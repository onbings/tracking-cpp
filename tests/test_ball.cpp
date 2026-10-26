/**
 * @file test_ball.cpp
 * @brief Tests for ball candidate filtering and the single-ball tracker gating.
 */
#include "ball.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

using namespace ft;

TEST(Ball, CandidateFilterRejectsConesAndColour) {
    cv::Mat frame(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, {100, 100}, 10, {255, 255, 255}, cv::FILLED);      // white ball
    cv::ellipse(frame, {300, 300}, {20, 6}, 0, 0, 360, {0, 220, 255}, cv::FILLED);  // yellow disc cone
    cv::circle(frame, {300, 100}, 10, {0, 140, 255}, cv::FILLED);        // orange round thing
    std::vector<Detection> dets = {
        {cv::Rect2f(90, 90, 20, 20), 0.6f, kBallClass},
        {cv::Rect2f(280, 294, 40, 12), 0.6f, kBallClass},   // cone: wrong aspect
        {cv::Rect2f(290, 90, 20, 20), 0.3f, kBallClass}};   // orange: saturated
    const auto kept = ballCandidates(dets, frame);
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_FLOAT_EQ(kept[0].box.x, 90.f);
}

TEST(Ball, TrackerPrefersNearCandidate) {
    cv::Mat frame(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, {100, 100}, 10, {255, 255, 255}, cv::FILLED);
    BallTracker bt(30.0);
    const std::vector<Detection> first = {{cv::Rect2f(90, 90, 20, 20), 0.6f, kBallClass}};
    ASSERT_TRUE(bt.update(first));
    EXPECT_TRUE(bt.reacquired());

    // A more confident far candidate and a close one: the gate must keep the close one.
    const std::vector<Detection> second = {
        {cv::Rect2f(190, 90, 20, 20), 0.9f, kBallClass},   // far
        {cv::Rect2f(95, 92, 20, 20), 0.5f, kBallClass}};   // near
    ASSERT_TRUE(bt.update(second));
    EXPECT_NEAR(bt.position().x, 105.0, 3.0);
    EXPECT_FALSE(bt.reacquired());
}
