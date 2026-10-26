/**
 * @file test_kinematics.cpp
 * @brief Tests that speed/distance are correct and that camera motion is compensated.
 */
#include "geometry.hpp"
#include "kinematics.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace ft;

namespace {
GroundModel makeGround() {
    GroundModel g(3840, 2160, 70, 1.4, 80);
    g.setParams(25.0 * CV_PI / 180.0, 6.0);
    return g;
}
}  // namespace

TEST(Kinematics, SpeedAndDistance) {
    const double fps = 30.0;
    const auto g = makeGround();
    Kinematics kin(fps, 2.0, 40.0, 15, 1.0, 0.5);
    const double speed = 5.0;  // m/s along X (18 km/h)
    const MotionState* s = nullptr;
    for (int i = 0; i < static_cast<int>(fps * 3); ++i) {
        bool valid = false;
        const cv::Point2d img = g.toImage({speed * i / fps - 7.0, 20.0}, valid);
        s = &kin.observe("p", img, g, i, 1.0);
    }
    ASSERT_NE(s, nullptr);
    EXPECT_NEAR(s->speedKmh(), 18.0, 2.0);
    EXPECT_NEAR(s->distance, speed * (3 - 1.0 / fps), 1.5);
    EXPECT_NEAR(s->maxSpeedKmh, 18.0, 3.0);
}

TEST(Kinematics, CompensatesCameraMotion) {
    const double fps = 30.0;
    const auto g = makeGround();
    Kinematics kin(fps, 4.0, 40.0, 15, 1.0, 0.5);
    const cv::Matx33d pan(1, 0, 15, 0, 1, 0, 0, 0, 1);  // camera pans: static player drifts in image
    bool valid = false;
    cv::Point2d img = g.toImage({0.0, 20.0}, valid);
    const MotionState* s = nullptr;
    for (int i = 0; i < 60; ++i) {
        kin.warp(pan);
        img += cv::Point2d(15, 0);  // image drift exactly matches the pan
        s = &kin.observe("p", img, g, i, 2.0);
    }
    ASSERT_NE(s, nullptr);
    EXPECT_LT(s->speedKmh(), 2.0);
    EXPECT_LT(s->distance, 1.0);
}
