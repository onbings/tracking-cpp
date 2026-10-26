/**
 * @file test_space.cpp
 * @brief Tests that Voronoi space control is symmetric and the radar renders.
 */
#include "geometry.hpp"
#include "space.hpp"

#include <gtest/gtest.h>

using namespace ft;

TEST(Space, SymmetricShare) {
    GroundModel g(3840, 2160, 70, 1.4, 80);
    g.setParams(30.0 * CV_PI / 180.0, 8.0);
    SpaceControl sc(g);
    const auto shares = sc.compute({{-5.0, 15.0}, {5.0, 15.0}}, {0, 1});
    ASSERT_TRUE(shares.has_value());
    ASSERT_EQ(shares->size(), 2u);
    EXPECT_NEAR(shares->at(0), 0.5, 0.05);

    std::map<int, cv::Vec3b> colors{{0, {0, 0, 255}}, {1, {0, 255, 255}}};
    const cv::Mat radar = sc.minimap({320, 240}, colors, {{{-5.0, 15.0}, {0, 0, 255}}}, std::nullopt);
    ASSERT_FALSE(radar.empty());
    EXPECT_EQ(radar.size(), cv::Size(320, 240));
}
