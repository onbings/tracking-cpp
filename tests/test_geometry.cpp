/**
 * @file test_geometry.cpp
 * @brief Tests for the ground model: projection round-trip, self-calibration, plausibility.
 */
#include "geometry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace ft;

namespace {
constexpr int W = 3840, H = 2160;

GroundModel makeGround(double tiltDeg = 25.0, double height = 6.0) {
    GroundModel g(W, H, 70, 1.4, 80);
    g.setParams(tiltDeg * CV_PI / 180.0, height);
    return g;
}

/// @brief Builds the box a `truth`-consistent person at ground (X,Y) would produce (optional noise).
cv::Rect2f personBox(const GroundModel& truth, cv::Point2d groundXY, double heightNoise = 0.0) {
    bool valid = false;
    const cv::Point2d foot = truth.toImage(groundXY, valid);
    const double hpx = truth.personBoxHeightPx(foot) * (1.0 + heightNoise);
    return {static_cast<float>(foot.x - hpx / 6), static_cast<float>(foot.y - hpx),
            static_cast<float>(hpx / 3), static_cast<float>(hpx)};
}
}  // namespace

TEST(Geometry, Roundtrip) {
    auto g = makeGround();
    for (cv::Point2d p : {cv::Point2d(100, 2000), cv::Point2d(1920, 1080), cv::Point2d(3500, 900)}) {
        bool v1 = false;
        const cv::Point2d ground = g.toGround(p, v1);
        ASSERT_TRUE(v1);
        bool v2 = false;
        const cv::Point2d back = g.toImage(ground, v2);
        ASSERT_TRUE(v2);
        EXPECT_NEAR(back.x, p.x, 1e-4);
        EXPECT_NEAR(back.y, p.y, 1e-4);
    }
}

TEST(Geometry, SelfCalibrationRecoversCamera) {
    const GroundModel truth = makeGround(22.0, 7.0);
    GroundModel fit(W, H, 70, 1.4, 80);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> ux(-15, 15), uy(8, 40);
    std::normal_distribution<double> noise(0, 0.03);
    for (int frame = 0; frame < 12; ++frame) {
        std::vector<cv::Rect2f> boxes;
        for (int i = 0; i < 15; ++i) {
            const cv::Rect2f b = personBox(truth, {ux(rng), uy(rng)}, noise(rng));
            if (b.height >= 24 && b.y > 4) boxes.push_back(b);
        }
        fit.addObservations(boxes);
    }
    for (int i = 0; i < 60; i += 5) fit.update(i);
    ASSERT_TRUE(fit.ready());
    EXPECT_NEAR(fit.tiltDeg(), 22.0, 4.0);
    EXPECT_NEAR(fit.camHeight(), 7.0, 1.5);
}

TEST(Geometry, PlausibilityFilters) {
    auto g = makeGround();
    const cv::Rect2f real = personBox(g, {0, 15});
    cv::Rect2f banner = real;
    banner.y -= 4 * real.height;
    banner.height = 5 * real.height;
    EXPECT_TRUE(g.plausiblePerson(real));
    EXPECT_FALSE(g.plausiblePerson(banner));
    EXPECT_GT(g.expectedSizePx(real.tl() + cv::Point2f(real.width / 2, real.height), 0.22), 0.0);
}

TEST(Geometry, VisibleBounds) {
    auto g = makeGround(30.0, 8.0);
    const auto bounds = g.visibleBounds();
    ASSERT_TRUE(bounds.has_value());
    EXPECT_LT((*bounds)[0], (*bounds)[2]);
    EXPECT_LT((*bounds)[1], (*bounds)[3]);
}
