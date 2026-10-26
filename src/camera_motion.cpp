/**
 * @file camera_motion.cpp
 * @brief Implementation of the turf-restricted KLT/RANSAC camera motion estimator.
 */
#include "camera_motion.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>

namespace ft {

cv::Mat grassMask(const cv::Mat& bgr) {
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    // OpenCV hue is 0..179; green turf under indoor light sits in [30, 95].
    cv::inRange(hsv, cv::Scalar(30, 40, 25), cv::Scalar(95, 255, 255), mask);
    return mask;
}

CameraMotion::CameraMotion(cv::Size fullSize, int procWidth)
    : fullSize_(fullSize),
      scale_(std::min(1.0, procWidth / static_cast<double>(fullSize.width))) {
    procSize_ = {static_cast<int>(std::lround(fullSize.width * scale_)),
                 static_cast<int>(std::lround(fullSize.height * scale_))};
}

cv::Point2d CameraMotion::warpPoint(const cv::Matx33d& h, const cv::Point2d& p) {
    const cv::Vec3d q = h * cv::Vec3d(p.x, p.y, 1.0);
    return {q[0] / q[2], q[1] / q[2]};
}

std::vector<cv::Point2d> CameraMotion::warpPoints(const cv::Matx33d& h, const std::vector<cv::Point2d>& pts) {
    std::vector<cv::Point2d> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.push_back(warpPoint(h, p));
    return out;
}

const cv::Matx33d& CameraMotion::update(const cv::Mat& image, const std::vector<cv::Rect2f>& excludeBoxes) {
    cv::Mat small, gray;
    cv::resize(image, small, procSize_, 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    grass_ = grassMask(small);

    // Feature mask: eroded turf, minus padded player boxes (players move independently of the camera).
    cv::Mat mask;
    cv::erode(grass_, mask, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    const double s = scale_;
    for (const auto& b : excludeBoxes) {
        cv::Rect r(cvRound(b.x * s) - 4, cvRound(b.y * s) - 4,
                   cvRound(b.width * s) + 8, cvRound(b.height * s) + 8);
        cv::rectangle(mask, r & cv::Rect(0, 0, mask.cols, mask.rows), cv::Scalar(0), cv::FILLED);
    }

    homography_ = cv::Matx33d::eye();
    inliers_ = 0;
    ok_ = false;
    cut_ = false;
    turfRatio_ = static_cast<double>(cv::countNonZero(grass_)) / static_cast<double>(grass_.total());
    // A camera filming the pitch has turf under it: the lower half is almost all grass (>= 0.95 on the
    // reference drone video), while close-ups of walls/banners/crowds stay far below even when the
    // banners are green (0.58 on a banner close-up).
    const cv::Mat lower = grass_.rowRange(grass_.rows / 2, grass_.rows);
    lowerTurfRatio_ = static_cast<double>(cv::countNonZero(lower)) / static_cast<double>(lower.total());

    // Hue/saturation histogram: robust to camera motion (same colours), sensitive to a new shot.
    cv::Mat hsv, hist;
    cv::cvtColor(small, hsv, cv::COLOR_BGR2HSV);
    const int channels[] = {0, 1};
    const int bins[] = {30, 32};
    const float hRange[] = {0, 180}, sRange[] = {0, 256};
    const float* ranges[] = {hRange, sRange};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, bins, ranges);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    histCorr_ = prevHist_.empty() ? 1.0 : cv::compareHist(prevHist_, hist, cv::HISTCMP_CORREL);
    prevHist_ = hist;

    if (histCorr_ < kCutCorrelation) {
        // Hard cut: the previous frame belongs to another shot, so feature tracking is meaningless.
        cut_ = true;
    } else if (!prevGray_.empty()) {
        std::vector<cv::Point2f> p0;
        cv::goodFeaturesToTrack(prevGray_, p0, maxCorners_, 0.01, 6, prevMask_, 5);
        if (p0.size() >= 12) {
            std::vector<cv::Point2f> p1;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prevGray_, gray, p0, p1, status, err, {21, 21}, 3);
            std::vector<cv::Point2f> a, b;
            for (std::size_t i = 0; i < status.size(); ++i) {
                if (status[i]) { a.push_back(p0[i]); b.push_back(p1[i]); }
            }
            if (a.size() >= 12) {
                std::vector<uchar> inlierMask;
                cv::Mat hSmall = cv::findHomography(a, b, cv::RANSAC, 1.5, inlierMask);
                const int nInliers = static_cast<int>(std::ranges::count(inlierMask, 1));
                if (!hSmall.empty() && nInliers >= 10) {
                    // Rescale the homography from processing to full resolution: H = S^-1 * Hs * S.
                    const cv::Matx33d scaleM(s, 0, 0, 0, s, 0, 0, 0, 1);
                    const cv::Matx33d scaleInv(1 / s, 0, 0, 0, 1 / s, 0, 0, 0, 1);
                    homography_ = scaleInv * cv::Matx33d(hSmall.ptr<double>()) * scaleM;
                    inliers_ = nInliers;
                    ok_ = true;
                }
            }
        }
        // Soft cut: the turf could not be followed and the colours shifted too (e.g. a cut between two
        // wide shots of the same pitch). Plain KLT failure alone is just blur and is not a cut.
        if (!ok_ && histCorr_ < kSoftCutCorrelation) cut_ = true;
    }

    prevGray_ = gray;
    prevMask_ = mask;
    return homography_;
}

}  // namespace ft
