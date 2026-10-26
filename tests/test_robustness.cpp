/**
 * @file test_robustness.cpp
 * @brief Tests for the input-robustness checks used on arbitrary clips: shot-cut detection,
 *        non-pitch views, and the resets applied at a cut.
 */
#include "camera_motion.hpp"
#include "geometry.hpp"
#include "tracker.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

using namespace ft;

namespace {
/// @brief A textured frame of a single hue (OpenCV hue 0..179), so KLT has corners to follow.
cv::Mat texturedFrame(int hue, cv::Size size = {640, 360}) {
    cv::RNG rng(1234);
    cv::Mat value(size, CV_8U);
    rng.fill(value, cv::RNG::UNIFORM, 60, 255);
    cv::GaussianBlur(value, value, {5, 5}, 1.5);
    cv::Mat hsv;
    cv::merge(std::vector<cv::Mat>{cv::Mat(size, CV_8U, cv::Scalar(hue)), cv::Mat(size, CV_8U, cv::Scalar(200)), value}, hsv);
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

cv::Mat shifted(const cv::Mat& img, double dx, double dy) {
    cv::Mat out;
    const cv::Matx23d m(1, 0, dx, 0, 1, dy);
    cv::warpAffine(img, out, m, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}
}  // namespace

TEST(Robustness, PanningOverTurfIsNotACut) {
    CameraMotion cam({640, 360});
    const cv::Mat turf = texturedFrame(60);  // green
    cam.update(turf, {});
    cam.update(shifted(turf, 4, 2), {});
    EXPECT_FALSE(cam.cut());
    EXPECT_TRUE(cam.ok());
    EXPECT_TRUE(cam.pitchView());
    EXPECT_GT(cam.histCorrelation(), 0.9);
    EXPECT_GT(cam.turfRatio(), 0.9);
    EXPECT_GT(cam.lowerTurfRatio(), 0.9);
}

TEST(Robustness, GreenBannerAboveIsNotEnoughForAPitchView) {
    // Top half green (banner), bottom half grey (wall/stands): plenty of "turf" overall, none underfoot.
    cv::Mat frame = texturedFrame(60);
    frame.rowRange(frame.rows / 2, frame.rows).setTo(cv::Scalar(128, 128, 128));
    CameraMotion cam({640, 360});
    cam.update(frame, {});
    EXPECT_NEAR(cam.turfRatio(), 0.5, 0.05);
    EXPECT_LT(cam.lowerTurfRatio(), 0.05);
    EXPECT_FALSE(cam.pitchView());
}

TEST(Robustness, SceneChangeIsACutAndNotAPitchView) {
    CameraMotion cam({640, 360});
    cam.update(texturedFrame(60), {});
    cam.update(texturedFrame(115), {});  // blue "crowd/close-up" frame
    EXPECT_TRUE(cam.cut());
    EXPECT_FALSE(cam.ok());
    EXPECT_FALSE(cam.pitchView());
    EXPECT_LT(cam.histCorrelation(), CameraMotion::kCutCorrelation);
}

TEST(Robustness, TrackerResetIssuesFreshIds) {
    BotSortTracker tracker(0.3f, 30, 1);
    const cv::Matx33d I = cv::Matx33d::eye();
    const Detection d{cv::Rect2f(100, 100, 40, 120), 0.9f, kPersonClass};
    std::vector<Track> before;
    for (int i = 0; i < 3; ++i) before = tracker.update({d}, I);
    ASSERT_EQ(before.size(), 1u);

    tracker.reset();
    std::vector<Track> after;
    for (int i = 0; i < 3; ++i) after = tracker.update({d}, I);
    ASSERT_EQ(after.size(), 1u);
    EXPECT_NE(after[0].id, before[0].id);  // a post-cut player never inherits a pre-cut id
}

TEST(Robustness, GroundResetRequiresRecalibration) {
    GroundModel g(1920, 1080, 70.0, 1.4, 60.0);
    g.setParams(0.4, 10.0);
    ASSERT_TRUE(g.ready());
    g.reset();
    EXPECT_FALSE(g.ready());
    bool valid = true;
    (void)g.toGround({960, 900}, valid);
    EXPECT_FALSE(valid);  // no metric output until the new shot is calibrated
}
