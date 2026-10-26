/**
 * @file overlay.cpp
 * @brief Implementation of the drawing helpers (pure OpenCV).
 */
#include "overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace ft::overlay {

namespace {
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kGolden = 0.618033988749895;
}  // namespace

cv::Vec3b colorForId(int id) {
    const double hue = std::fmod(id * kGolden, 1.0) * 180.0;  // OpenCV hue 0..179
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 190, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

void darkenRect(cv::Mat& img, cv::Rect rect, double alpha) {
    rect &= cv::Rect(0, 0, img.cols, img.rows);
    if (rect.area() == 0) return;
    cv::Mat roi = img(rect);
    roi.convertTo(roi, -1, 1.0 - alpha, 0);
}

void drawLabel(cv::Mat& img, const std::string& text, int x, int y, const cv::Scalar& color, double scale) {
    int baseline = 0;
    const cv::Size sz = cv::getTextSize(text, kFont, scale, 1, &baseline);
    y = std::max(sz.height + baseline + 2, y);
    cv::rectangle(img, {x, y - sz.height - baseline - 4}, {x + sz.width + 6, y}, color, cv::FILLED);
    const double lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2];
    const cv::Scalar fg = lum > 140 ? cv::Scalar(0, 0, 0) : kWhite;
    cv::putText(img, text, {x + 3, y - baseline - 2}, kFont, scale, fg, 1, cv::LINE_AA);
}

void drawDashedRect(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Scalar& color, int dash) {
    for (int x = p1.x; x < p2.x; x += dash * 2) {
        cv::line(img, {x, p1.y}, {std::min(x + dash, p2.x), p1.y}, color, 1);
        cv::line(img, {x, p2.y}, {std::min(x + dash, p2.x), p2.y}, color, 1);
    }
    for (int y = p1.y; y < p2.y; y += dash * 2) {
        cv::line(img, {p1.x, y}, {p1.x, std::min(y + dash, p2.y)}, color, 1);
        cv::line(img, {p2.x, y}, {p2.x, std::min(y + dash, p2.y)}, color, 1);
    }
}

void drawTrail(cv::Mat& img, const std::vector<cv::Point2f>& points, const cv::Scalar& color) {
    const int n = static_cast<int>(points.size());
    for (int i = 1; i < n; ++i) {
        cv::line(img, points[i - 1], points[i], color, 1 + 3 * i / n, cv::LINE_AA);
    }
}

void drawSparkline(cv::Mat& img, const std::vector<double>& values, cv::Rect box, const cv::Scalar& color,
                   const std::string& label) {
    cv::rectangle(img, box, {80, 80, 80}, 1);
    if (values.size() < 2) return;
    const double vmax = std::max(1e-6, *std::ranges::max_element(values));
    std::vector<cv::Point> pts;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = box.x + 1 + (box.width - 2) * static_cast<double>(i) / (values.size() - 1);
        const double y = box.y + box.height - 1 - values[i] / vmax * (box.height - 2);
        pts.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
    cv::polylines(img, pts, false, color, 1, cv::LINE_AA);
    if (!label.empty()) {
        cv::putText(img, label + " max " + std::to_string(static_cast<int>(vmax)), {box.x + 4, box.y + 12},
                    kFont, 0.38, kGrey, 1, cv::LINE_AA);
    }
}

void drawStageBar(cv::Mat& img, const std::vector<std::pair<std::string, double>>& stages, cv::Rect box) {
    static const std::map<std::string, cv::Scalar> colors = {
        {"decode", {255, 170, 60}}, {"wait", {140, 140, 140}}, {"detect", {60, 90, 255}},
        {"camera", {0, 200, 255}}, {"track", {60, 220, 120}}, {"analytics", {255, 255, 0}},
        {"ball", {255, 255, 255}}, {"render", {230, 90, 230}}};
    cv::rectangle(img, box, {80, 80, 80}, 1);
    double total = 0;
    for (auto& [name, ms] : stages) total += ms;
    if (total <= 0) return;
    int cx = box.x;
    for (auto& [name, ms] : stages) {
        const int seg = static_cast<int>(std::lround(box.width * ms / total));
        if (seg <= 0) continue;
        auto it = colors.find(name);
        cv::rectangle(img, {cx, box.y + 1}, {std::min(cx + seg, box.x + box.width), box.y + box.height - 1},
                      it != colors.end() ? it->second : kGrey, cv::FILLED);
        if (seg > 40) cv::putText(img, name, {cx + 3, box.y + box.height - 4}, kFont, 0.36, {0, 0, 0}, 1, cv::LINE_AA);
        cx += seg;
    }
}

void drawHud(cv::Mat& img, const std::vector<std::pair<std::string, cv::Scalar>>& lines,
             const std::vector<double>& fpsHistory, const std::vector<std::pair<std::string, double>>& stageMs,
             double scale) {
    const int lineH = static_cast<int>(34 * scale) + 4;
    const int width = static_cast<int>(1150 * scale);
    const int graphH = static_cast<int>(70 * scale) + 10;
    const int barH = static_cast<int>(30 * scale) + 6;
    const int height = 12 + lineH * static_cast<int>(lines.size()) + graphH + barH + 22;
    darkenRect(img, {10, 10, width, height}, 0.65);
    int y = 10 + lineH;
    for (auto& [text, color] : lines) {
        cv::putText(img, text, {20, y}, kFont, scale, color, 1, cv::LINE_AA);
        y += lineH;
    }
    drawSparkline(img, fpsHistory, {20, y - lineH + 12, width - 20, graphH}, kHudGreen, "FPS");
    y += graphH + 6;
    drawStageBar(img, stageMs, {20, y - lineH + 12, width - 20, barH});
}

void drawPanel(cv::Mat& img, const std::vector<std::tuple<std::string, cv::Scalar, cv::Scalar>>& rows,
               cv::Point topRight, double scale) {
    const int lineH = static_cast<int>(34 * scale) + 4;
    const int sw = lineH - 6;
    int width = 0;
    for (auto& [text, color, swatch] : rows) {
        width = std::max(width, cv::getTextSize(text, kFont, scale, 1, nullptr).width);
    }
    width += 20 + sw + 6;
    const int height = lineH * static_cast<int>(rows.size()) + 10;
    const int x0 = topRight.x - width;
    darkenRect(img, {x0, topRight.y, width, height}, 0.65);
    int y = topRight.y + lineH;
    for (auto& [text, color, swatch] : rows) {
        int tx = x0 + 10;
        if (swatch[0] >= 0) {
            cv::rectangle(img, {tx, y - sw + 2}, {tx + sw, y + 2}, swatch, cv::FILLED);
        }
        tx += sw + 6;
        cv::putText(img, text, {tx, y}, kFont, scale, color, 1, cv::LINE_AA);
        y += lineH;
    }
}

void drawBall(cv::Mat& img, cv::Point center, int radius, const std::vector<cv::Point2f>& trail,
              const std::string& text, bool predicted) {
    const int n = static_cast<int>(trail.size());
    for (int i = 1; i < n; ++i) {
        cv::line(img, trail[i - 1], trail[i], {0, 255, 255}, 1 + 3 * i / n, cv::LINE_AA);
    }
    const cv::Scalar color = predicted ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
    cv::circle(img, center, std::max(radius + 6, 9), color, 2, cv::LINE_AA);
    cv::drawMarker(img, center, color, cv::MARKER_CROSS, 8, 1, cv::LINE_AA);
    drawLabel(img, text, center.x + radius + 8, center.y - radius - 4, color, 0.45);
}

void drawCountingLine(cv::Mat& img, cv::Point2d a, cv::Point2d b, double scale, bool flash, const std::string& text) {
    const cv::Point pa(static_cast<int>(a.x * scale), static_cast<int>(a.y * scale));
    const cv::Point pb(static_cast<int>(b.x * scale), static_cast<int>(b.y * scale));
    const cv::Scalar color = flash ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 0);
    cv::line(img, pa, pb, color, flash ? 4 : 2, cv::LINE_AA);
    drawLabel(img, text, (pa.x + pb.x) / 2 - 60, (pa.y + pb.y) / 2 - 6, color, 0.55);
}

void drawBanner(cv::Mat& img, const std::string& text) {
    int baseline = 0;
    const cv::Size sz = cv::getTextSize(text, kFont, 1.2, 2, &baseline);
    const int x = (img.cols - sz.width) / 2, y = img.rows / 2;
    darkenRect(img, {x - 20, y - sz.height - 20, sz.width + 40, sz.height + 40}, 0.6);
    cv::putText(img, text, {x, y}, kFont, 1.2, {0, 255, 255}, 2, cv::LINE_AA);
}

}  // namespace ft::overlay
