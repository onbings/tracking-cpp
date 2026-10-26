/**
 * @file test_team.cpp
 * @brief Tests for jersey colour naming and team clustering with outliers.
 */
#include "team.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ft;

TEST(Team, ColorNames) {
    EXPECT_EQ(colorName(cv::Vec3b(0, 0, 255)), "red");
    EXPECT_EQ(colorName(cv::Vec3b(0, 255, 255)), "yellow");
    EXPECT_EQ(colorName(cv::Vec3b(255, 255, 255)), "white");
}

TEST(Team, ClustersWithOutliers) {
    std::mt19937 rng(3);
    std::normal_distribution<float> n(0, 4);
    const cv::Vec3f red(110, 165, 150), yellow(200, 120, 175);
    TeamClassifier tc(60);
    for (int i = 0; i < 80; ++i) {
        const int id = i % 8;
        const cv::Vec3f base = id < 4 ? red : yellow;
        tc.observe(id, cv::Vec3f(base[0] + n(rng), base[1] + n(rng), base[2] + n(rng)));
    }
    ASSERT_TRUE(tc.ready());
    const auto t0 = tc.teamOf(0);
    const auto t4 = tc.teamOf(4);
    ASSERT_TRUE(t0.has_value());
    ASSERT_TRUE(t4.has_value());
    EXPECT_NE(*t0, *t4);

    for (int i = 0; i < 5; ++i) tc.observe(99, cv::Vec3f(225, 129, 130));  // white shirt
    EXPECT_EQ(tc.teamOf(99), kTeamOther);
}
