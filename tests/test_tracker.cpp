/**
 * @file test_tracker.cpp
 * @brief Tests for the BoT-SORT-style tracker and the Kalman box.
 */
#include "tracker.hpp"

#include <gtest/gtest.h>

using namespace ft;

namespace {
Detection person(float x, float y, float w, float h, float score, int id = kPersonClass) {
    return {cv::Rect2f(x, y, w, h), score, id};
}
}  // namespace

TEST(Tracker, ConfirmsAndKeepsStableId) {
    BotSortTracker tracker(0.3f, 30, 3);
    const cv::Matx33d I = cv::Matx33d::eye();
    int id = -1;
    for (int frame = 0; frame < 6; ++frame) {
        auto tracks = tracker.update({person(100 + frame * 5, 100, 40, 120, 0.9f)}, I);
        if (frame >= 2) {  // confirmed after nInit hits
            ASSERT_EQ(tracks.size(), 1u);
            if (id < 0) id = tracks[0].id;
            EXPECT_EQ(tracks[0].id, id);  // id must not change
        }
    }
    EXPECT_GE(id, 1);
}

TEST(Tracker, SeparatesTwoObjects) {
    BotSortTracker tracker(0.3f, 30, 1);
    const cv::Matx33d I = cv::Matx33d::eye();
    for (int frame = 0; frame < 4; ++frame) {
        tracker.update({person(100, 100, 40, 120, 0.9f), person(900, 500, 40, 120, 0.9f)}, I);
    }
    auto tracks = tracker.update({person(100, 100, 40, 120, 0.9f), person(900, 500, 40, 120, 0.9f)}, I);
    ASSERT_EQ(tracks.size(), 2u);
    EXPECT_NE(tracks[0].id, tracks[1].id);
}

TEST(Tracker, DropsStaleTracks) {
    BotSortTracker tracker(0.3f, 3, 1);
    const cv::Matx33d I = cv::Matx33d::eye();
    tracker.update({person(100, 100, 40, 120, 0.9f)}, I);
    for (int frame = 0; frame < 6; ++frame) tracker.update({}, I);  // no detections
    auto tracks = tracker.update({}, I);
    EXPECT_TRUE(tracks.empty());
}
