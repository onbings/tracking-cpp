/**
 * @file ball.cpp
 * @brief Implementation of ball candidate filtering and the single-ball tracker.
 */
#include "ball.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace ft {

std::vector<Detection> ballCandidates(const std::vector<Detection>& dets, const cv::Mat& frame) {
    std::vector<Detection> kept;
    for (const auto& d : dets) {
        const float w = d.box.width, h = d.box.height;
        // Shape: roughly square and small. Disc cones come out at aspect 1.9-3.2.
        if (d.score < 0.12f || w < 3 || h < 3 || w > 0.08f * frame.cols) continue;
        const float aspect = w / h;
        if (aspect < 0.6f || aspect > 1.6f) continue;
        // Colour: a match ball is white/black; cones are vivid. Looser limit for confident detections
        // because indoor light tints the ball.
        const float cx = d.box.x + w * 0.5f, cy = d.box.y + h * 0.5f, r = std::max(1.0f, std::min(w, h) * 0.3f);
        cv::Rect patch(cvRound(cx - r), cvRound(cy - r), cvRound(2 * r + 1), cvRound(2 * r + 1));
        patch &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (patch.area() == 0) continue;
        cv::Mat hsv;
        cv::cvtColor(frame(patch), hsv, cv::COLOR_BGR2HSV);
        cv::Scalar meanHsv = cv::mean(hsv);
        const double maxSat = d.score >= 0.4f ? 160.0 : 120.0;
        if (meanHsv[1] <= maxSat) kept.push_back(d);
    }
    return kept;
}

std::vector<Detection> plausibleBallSize(const std::vector<Detection>& cands, const GroundModel& ground) {
    if (!ground.ready()) return cands;
    std::vector<Detection> kept;
    for (const auto& d : cands) {
        const double size = std::max(d.box.width, d.box.height);
        const cv::Point2d bottom(d.box.x + d.box.width * 0.5, d.box.y + d.box.height);
        const double expected = ground.expectedSizePx(bottom, 0.22);  // 22 cm ball
        if (expected <= 0) { kept.push_back(d); continue; }
        const double ratio = size / expected;
        if (ratio >= 0.4 && ratio <= 2.5) kept.push_back(d);  // generous upper bound: airborne balls look bigger
    }
    return kept;
}

BallTracker::BallTracker(double fps, int trailLength)
    : maxLost_(static_cast<int>(fps * 0.8)), trailLength_(static_cast<std::size_t>(trailLength)) {}

void BallTracker::warp(const cv::Matx33d& h) {
    auto apply = [&](cv::Point2d& p) {
        const cv::Vec3d q = h * cv::Vec3d(p.x, p.y, 1.0);
        p = {q[0] / q[2], q[1] / q[2]};
    };
    if (pos_) apply(*pos_);
    for (auto& p : trail_) apply(p);
}

std::optional<cv::Point2d> BallTracker::predicted() const {
    if (!pos_) return std::nullopt;
    return *pos_ + vel_;
}

bool BallTracker::update(const std::vector<Detection>& candidates) {
    reacquired_ = false;
    const auto pred = predicted();
    int best = -1;

    if (!candidates.empty()) {
        if (pred) {
            // Gate relative to the apparent ball size (several balls may be nearby).
            const double gate = std::max(4.0 * size_, 0.4 * gatePx_) * (1 + 0.3 * lost_);
            double bestScore = -1e9;
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                const cv::Point2d c = candidates[i].center();
                const double dist = cv::norm(c - *pred);
                if (dist > gate) continue;
                const double score = candidates[i].score - 0.5 * dist / gate;
                if (score > bestScore) { bestScore = score; best = i; }
            }
            if (best >= 0 && cv::norm(candidates[best].center() - *pred) > 2.5 * size_ * (1 + lost_)) {
                reacquired_ = true;  // plausible but large jump: don't integrate it as motion
                trail_.clear();
            }
        }
        if (best < 0 && (!pred || lost_ > 2)) {
            // Re-acquire: take the most confident candidate anywhere.
            int j = 0;
            for (int i = 1; i < static_cast<int>(candidates.size()); ++i) {
                if (candidates[i].score > candidates[j].score) j = i;
            }
            if (candidates[j].score >= acquireConf_) {
                best = j;
                vel_ = {0, 0};
                hits_ = 0;
                trail_.clear();
                reacquired_ = true;
            }
        }
    }

    if (best < 0) {
        if (pos_) {
            lost_ += 1;
            pos_ = pred;
            vel_ *= 0.9;
            if (lost_ > maxLost_) reset();
        }
        return false;
    }

    const cv::Point2d center = candidates[best].center();
    if (pos_ && hits_ > 0) vel_ = 0.5 * vel_ + 0.5 * (center - *pos_) / (1 + lost_);
    pos_ = center;
    conf_ = candidates[best].score;
    size_ = 0.7 * size_ + 0.3 * std::max(candidates[best].box.width, candidates[best].box.height);
    lost_ = 0;
    hits_ += 1;
    detectedFrames_ += 1;
    trail_.push_back(center);
    while (trail_.size() > trailLength_) trail_.pop_front();
    return true;
}

bool BallTracker::visible() const { return pos_.has_value() && lost_ == 0 && hits_ >= 2; }

std::optional<cv::Point2d> BallTracker::groundPoint() const {
    if (!pos_) return std::nullopt;
    return cv::Point2d(pos_->x, pos_->y + size_ * 0.5);
}

std::optional<cv::Rect> BallTracker::roi(cv::Size frameSize, int size) const {
    const auto p = predicted();
    if (!p) return std::nullopt;
    const int half = size / 2;
    const int x0 = std::clamp(static_cast<int>(p->x) - half, 0, std::max(0, frameSize.width - size));
    const int y0 = std::clamp(static_cast<int>(p->y) - half, 0, std::max(0, frameSize.height - size));
    return cv::Rect(x0, y0, std::min(size, frameSize.width - x0), std::min(size, frameSize.height - y0));
}

void BallTracker::reset() {
    pos_.reset();
    vel_ = {0, 0};
    lost_ = 0;
    hits_ = 0;
    trail_.clear();
}

}  // namespace ft
