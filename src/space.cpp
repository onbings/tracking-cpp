/**
 * @file space.cpp
 * @brief Implementation of Voronoi space control and the radar minimap.
 */
#include "space.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace ft {

namespace {
constexpr int kNoData = -2;
}

SpaceControl::SpaceControl(const GroundModel& ground) : ground_(ground) {}

void SpaceControl::setTurf(const cv::Mat& grass, double scale) {
    const int k = std::max(3, (static_cast<int>(std::lround(grass.cols / 50.0)) | 1));
    cv::morphologyEx(grass, turf_, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
    turfScale_ = scale;
}

bool SpaceControl::onTurf(const cv::Point2d& imagePt) const {
    if (turf_.empty()) return true;
    const int u = static_cast<int>(imagePt.x * turfScale_);
    const int v = static_cast<int>(imagePt.y * turfScale_);
    if (u < 0 || u >= turf_.cols || v < 0 || v >= turf_.rows) return false;
    return turf_.at<uchar>(v, u) > 0;
}

std::optional<std::map<int, double>> SpaceControl::compute(const std::vector<cv::Point2d>& positions,
                                                           const std::vector<int>& teams) {
    hasRaster_ = false;
    shares_.clear();
    const auto bounds = ground_.visibleBounds();
    if (!bounds || positions.empty()) return std::nullopt;

    const double xmin = (*bounds)[0], ymin = (*bounds)[1], xmax = (*bounds)[2], ymax = (*bounds)[3];
    const double area = std::max(1e-6, (xmax - xmin) * (ymax - ymin));
    const double step = std::max(step_, std::sqrt(area / maxCells_));
    const int nx = std::max(1, static_cast<int>((xmax - xmin) / step));
    const int ny = std::max(1, static_cast<int>((ymax - ymin) / step));

    raster_ = cv::Mat(ny, nx, CV_16S, cv::Scalar(kNoData));
    std::map<int, long> counts;
    long total = 0;
    for (int gy = 0; gy < ny; ++gy) {
        for (int gx = 0; gx < nx; ++gx) {
            const cv::Point2d g(xmin + (gx + 0.5) * step, ymin + (gy + 0.5) * step);
            bool valid = false;
            const cv::Point2d img = ground_.toImage(g, valid);
            if (!valid) continue;
            if (!onTurf(img)) continue;
            // Nearest player.
            int ownerTeam = kNoData;
            double bestD = 1e18;
            for (std::size_t i = 0; i < positions.size(); ++i) {
                const double d = cv::norm(positions[i] - g);
                if (d < bestD) { bestD = d; ownerTeam = teams[i]; }
            }
            raster_.at<int16_t>(gy, gx) = static_cast<int16_t>(ownerTeam);
            counts[ownerTeam] += 1;
            total += 1;
        }
    }
    if (total == 0) return std::nullopt;
    bounds_ = cv::Vec4d(xmin, ymin, xmin + nx * step, ymin + ny * step);
    hasRaster_ = true;
    for (auto [team, n] : counts) shares_[team] = static_cast<double>(n) / total;
    return shares_;
}

void SpaceControl::overlay(cv::Mat& canvas, const std::vector<cv::Point2d>& positions,
                           const std::vector<int>& teams, const std::map<int, cv::Vec3b>& colors) const {
    if (positions.empty() || !ground_.ready()) return;
    const int cell = 10;
    // The ground model works in full-resolution pixels; map canvas pixels to full resolution.
    const double fullOverCanvas = static_cast<double>(ground_.width()) / canvas.cols;

    cv::Mat tint = canvas.clone();
    for (int y = 0; y < canvas.rows; y += cell) {
        for (int x = 0; x < canvas.cols; x += cell) {
            const cv::Point2d img((x + cell / 2.0) * fullOverCanvas, (y + cell / 2.0) * fullOverCanvas);
            bool valid = false;
            const cv::Point2d g = ground_.toGround(img, valid);
            if (!valid || !onTurf(img)) continue;
            int ownerTeam = kNoData;
            double bestD = 1e18;
            for (std::size_t i = 0; i < positions.size(); ++i) {
                const double d = cv::norm(positions[i] - g);
                if (d < bestD) { bestD = d; ownerTeam = teams[i]; }
            }
            auto it = colors.find(ownerTeam);
            if (it == colors.end()) continue;
            cv::rectangle(tint, cv::Rect(x, y, cell, cell),
                          cv::Scalar(it->second[0], it->second[1], it->second[2]), cv::FILLED);
        }
    }
    cv::addWeighted(canvas, 0.65, tint, 0.35, 0, canvas);
}

cv::Mat SpaceControl::minimap(cv::Size size, const std::map<int, cv::Vec3b>& colors,
                              const std::vector<std::pair<cv::Point2d, cv::Vec3b>>& players,
                              const std::optional<cv::Point2d>& ball) const {
    if (!hasRaster_) return {};
    const double xmin = bounds_[0], ymin = bounds_[1], xmax = bounds_[2], ymax = bounds_[3];
    const double scale = std::min((size.width - 10) / (xmax - xmin), (size.height - 10) / (ymax - ymin));
    const int ny = raster_.rows, nx = raster_.cols;

    cv::Mat img(ny, nx, CV_8UC3, cv::Scalar(40, 40, 40));
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            auto it = colors.find(raster_.at<int16_t>(y, x));
            if (it != colors.end()) {
                const cv::Vec3b c = it->second;
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(c[0] * 0.45 + 40 * 0.55),
                    static_cast<uchar>(c[1] * 0.45 + 90 * 0.55),
                    static_cast<uchar>(c[2] * 0.45 + 40 * 0.55));
            }
        }
    }
    const int outW = std::max(1, static_cast<int>((xmax - xmin) * scale));
    const int outH = std::max(1, static_cast<int>((ymax - ymin) * scale));
    cv::Mat flipped;
    cv::flip(img, flipped, 0);  // depth increases upwards on the radar
    cv::resize(flipped, flipped, {outW, outH}, 0, 0, cv::INTER_NEAREST);

    cv::Mat canvas(size, CV_8UC3, cv::Scalar(25, 25, 25));
    const int ox = (size.width - outW) / 2, oy = (size.height - outH) / 2;
    flipped.copyTo(canvas(cv::Rect(ox, oy, outW, outH)));

    auto toPx = [&](const cv::Point2d& p) {
        return cv::Point(static_cast<int>(ox + (p.x - xmin) * scale),
                         static_cast<int>(oy + outH - (p.y - ymin) * scale));
    };
    for (double gx = std::ceil(xmin / 10) * 10; gx < xmax; gx += 10) {
        const int px = toPx({gx, ymin}).x;
        cv::line(canvas, {px, oy}, {px, oy + outH}, {70, 70, 70}, 1);
    }
    for (double gy = std::ceil(ymin / 10) * 10; gy < ymax; gy += 10) {
        const int py = toPx({xmin, gy}).y;
        cv::line(canvas, {ox, py}, {ox + outW, py}, {70, 70, 70}, 1);
    }
    for (const auto& [p, c] : players) {
        cv::circle(canvas, toPx(p), 5, cv::Scalar(c[0], c[1], c[2]), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, toPx(p), 5, {0, 0, 0}, 1, cv::LINE_AA);
    }
    if (ball) {
        cv::circle(canvas, toPx(*ball), 4, {255, 255, 255}, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, toPx(*ball), 6, {0, 0, 255}, 1, cv::LINE_AA);
    }
    cv::putText(canvas, "radar (10 m grid)", {6, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200, 200, 200}, 1, cv::LINE_AA);
    cv::rectangle(canvas, cv::Rect(0, 0, size.width - 1, size.height - 1), {120, 120, 120}, 1);
    return canvas;
}

}  // namespace ft
