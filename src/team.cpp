/**
 * @file team.cpp
 * @brief Implementation of jersey-colour team classification.
 */
#include "team.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ft {

namespace {
// Lab feature weights: lightness (shadows, lighting) matters less than chroma for jerseys.
const cv::Vec3f kWeights(0.35f, 1.0f, 1.0f);
constexpr float kVoteDecay = 0.98f;
constexpr float kMinVotes = 3.0f;

/// @return chroma sqrt((a-128)^2 + (b-128)^2) of a Lab colour.
float chroma(const cv::Vec3f& lab) {
    return std::hypot(lab[1] - 128.0f, lab[2] - 128.0f);
}

/// @brief Weighted Euclidean distance in Lab.
float weightedDist(const cv::Vec3f& a, const cv::Vec3f& b) {
    cv::Vec3f d = a - b;
    return std::sqrt((d[0] * d[0]) * kWeights[0] * kWeights[0] +
                     (d[1] * d[1]) * kWeights[1] * kWeights[1] +
                     (d[2] * d[2]) * kWeights[2] * kWeights[2]);
}

/// @brief Boosts saturation/brightness of a muddy median jersey colour for drawing.
cv::Vec3b vivid(const cv::Vec3b& bgr) {
    cv::Mat hsv;
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2])), hsv, cv::COLOR_BGR2HSV);
    cv::Vec3b& p = hsv.at<cv::Vec3b>(0, 0);
    if (p[1] > 50) { p[1] = 230; p[2] = std::max<uchar>(p[2], 240); }
    else { p[2] = p[2] > 128 ? 240 : 70; }
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out.at<cv::Vec3b>(0, 0);
}
}  // namespace

std::string colorName(const cv::Vec3b& bgr) {
    cv::Mat hsv;
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2])), hsv, cv::COLOR_BGR2HSV);
    const cv::Vec3b p = hsv.at<cv::Vec3b>(0, 0);
    const double h = p[0] * 2.0, s = p[1] / 255.0, v = p[2] / 255.0;
    if (v < 0.25) return "black";
    if (s < 0.22) return v > 0.7 ? "white" : "grey";
    static const std::pair<double, const char*> hues[] = {
        {15, "red"}, {40, "orange"}, {70, "yellow"}, {160, "green"},
        {200, "cyan"}, {255, "blue"}, {290, "purple"}, {335, "pink"}, {361, "red"}};
    for (auto [limit, name] : hues) if (h < limit) return name;
    return "?";
}

std::optional<cv::Vec3f> jerseyFeature(const cv::Mat& frame, const cv::Rect2f& box) {
    const float w = box.width, h = box.height;
    if (w < 6 || h < 16) return std::nullopt;
    // Torso region: 20-50% of the height, central 60% of the width.
    cv::Rect roi(cvRound(box.x + 0.2f * w), cvRound(box.y + 0.2f * h), cvRound(0.6f * w), cvRound(0.3f * h));
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (roi.width < 3 || roi.height < 3) return std::nullopt;

    cv::Mat crop;
    cv::resize(frame(roi), crop, {12, 16}, 0, 0, cv::INTER_AREA);
    cv::Mat hsv, lab;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(crop, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Vec3f> samples;
    for (int y = 0; y < crop.rows; ++y) {
        for (int x = 0; x < crop.cols; ++x) {
            const cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
            const bool grass = p[0] >= 30 && p[0] <= 95 && p[1] >= 40 && p[2] >= 25;  // turf shows between arms
            if (!grass) samples.push_back(lab.at<cv::Vec3b>(y, x));
        }
    }
    if (samples.size() < 24) return std::nullopt;
    cv::Vec3f median;
    for (int c = 0; c < 3; ++c) {
        std::vector<float> ch(samples.size());
        std::ranges::transform(samples, ch.begin(), [c](const cv::Vec3f& s) { return s[c]; });
        std::ranges::nth_element(ch, ch.begin() + ch.size() / 2);
        median[c] = ch[ch.size() / 2];
    }
    return median;
}

TeamClassifier::TeamClassifier(int warmupSamples) : warmupSamples_(warmupSamples) {}

void TeamClassifier::observe(int trackId, const std::optional<cv::Vec3f>& feature) {
    if (!feature) return;
    if (!ready_) {
        pending_.emplace_back(trackId, *feature);
        std::vector<int> distinct;
        for (auto& [id, f] : pending_) if (std::ranges::find(distinct, id) == distinct.end()) distinct.push_back(id);
        if (static_cast<int>(pending_.size()) >= warmupSamples_ && static_cast<int>(distinct.size()) >= 2 * numTeams_) {
            fit();
        }
        return;
    }
    auto& v = votes_[trackId];
    for (float& value : v) value *= kVoteDecay;
    const int team = classify(*feature);
    v[team == kTeamOther ? 2 : team] += 1.0f;
}

int TeamClassifier::classify(const cv::Vec3f& feature) const {
    const int j = weightedDist(centers_[0], feature) <= weightedDist(centers_[1], feature) ? 0 : 1;
    if (weightedDist(centers_[j], feature) > threshold_) return kTeamOther;
    // A white/grey shirt near a saturated kit is not that team, however close in Lab.
    const float cTeam = chroma(centers_[j]), cObs = chroma(feature);
    if (cTeam > 15 && std::abs(cObs - cTeam) > 0.6f * cTeam) return kTeamOther;
    return j;
}

void TeamClassifier::fit() {
    cv::Mat data(static_cast<int>(pending_.size()), 3, CV_32F);
    for (int i = 0; i < data.rows; ++i) {
        for (int c = 0; c < 3; ++c) data.at<float>(i, c) = pending_[i].second[c] * kWeights[c];
    }
    cv::Mat labels, centers;
    cv::kmeans(data, numTeams_, labels,
               {cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.5}, 8, cv::KMEANS_PP_CENTERS, centers);

    std::array<cv::Vec3f, 2> raw;
    for (int k = 0; k < numTeams_; ++k) {
        for (int c = 0; c < 3; ++c) raw[k][c] = centers.at<float>(k, c) / kWeights[c];
    }
    // Outlier threshold from the spread of the samples around their centre.
    std::vector<float> dists(pending_.size());
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        dists[i] = weightedDist(pending_[i].second, raw[labels.at<int>(static_cast<int>(i))]);
    }
    std::ranges::nth_element(dists, dists.begin() + dists.size() / 2);
    threshold_ = std::max(12.0f, 2.0f * dists[dists.size() / 2]);

    // Order teams by hue so "A"/"B" are stable between runs.
    auto toBgr = [](const cv::Vec3f& lab) {
        cv::Mat m(1, 1, CV_8UC3, cv::Scalar(lab[0], lab[1], lab[2])), out;
        cv::cvtColor(m, out, cv::COLOR_Lab2BGR);
        return out.at<cv::Vec3b>(0, 0);
    };
    auto hueOf = [](const cv::Vec3b& bgr) {
        cv::Mat m(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2])), hsv;
        cv::cvtColor(m, hsv, cv::COLOR_BGR2HSV);
        return hsv.at<cv::Vec3b>(0, 0)[0];
    };
    std::array<int, 2> order{0, 1};
    if (hueOf(toBgr(raw[0])) > hueOf(toBgr(raw[1]))) std::swap(order[0], order[1]);
    for (int k = 0; k < numTeams_; ++k) {
        centers_[k] = raw[order[k]];
        colors_[k] = vivid(toBgr(raw[order[k]]));
        names_[k] = std::string(1, static_cast<char>('A' + k)) + " (" + colorName(colors_[k]) + ")";
    }
    ready_ = true;
    for (auto& [id, f] : pending_) {
        auto& v = votes_[id];
        const int team = classify(f);
        v[team == kTeamOther ? 2 : team] += 1.0f;
    }
    pending_.clear();
}

std::optional<int> TeamClassifier::teamOf(int trackId) const {
    auto it = votes_.find(trackId);
    if (it == votes_.end()) return std::nullopt;
    const auto& v = it->second;
    if (v[0] + v[1] + v[2] < kMinVotes) return std::nullopt;
    const int arg = static_cast<int>(std::distance(v.begin(), std::ranges::max_element(v)));
    return arg == 2 ? kTeamOther : arg;
}

cv::Vec3b TeamClassifier::color(int team) const {
    if (team == kTeamOther || team < 0 || team >= numTeams_) return {200, 200, 200};
    return colors_[team];
}

std::string TeamClassifier::name(int team) const {
    if (team == kTeamOther) return "other";
    if (team < 0 || team >= numTeams_) return "?";
    return names_[team];
}

std::string TeamClassifier::label(std::optional<int> team) const {
    if (!team) return "?";
    if (*team == kTeamOther) return "-";
    return std::string(1, static_cast<char>('A' + *team));
}

}  // namespace ft
