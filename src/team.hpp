/**
 * @file team.hpp
 * @brief Unsupervised team assignment from jersey colour.
 *
 * The median Lab colour of each player's torso (turf pixels removed) is clustered into two teams by
 * k-means. Each track's team is a decayed majority vote. Distance and chroma tests send referees,
 * coaches and staff to "other". No training data is required. See DOCUMENTATION.md §16.
 */
#pragma once

#include "common.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ft {

/// @return a human colour name ("red", "yellow", ...) for a BGR colour.
std::string colorName(const cv::Vec3b& bgr);

/**
 * @brief Extracts the median Lab colour of a player's torso.
 * @param frame BGR frame.
 * @param box   player box in image pixels.
 * @return the Lab feature, or std::nullopt if the crop is unusable (too small, all turf).
 */
std::optional<cv::Vec3f> jerseyFeature(const cv::Mat& frame, const cv::Rect2f& box);

/**
 * @brief Clusters jersey colours into two teams and classifies tracks by voting.
 */
class TeamClassifier {
public:
    /// @brief Builds a classifier that fits after `warmupSamples` features from several tracks.
    explicit TeamClassifier(int warmupSamples = 120);

    /**
     * @brief Records a feature for a track (clusters once enough have accumulated).
     * @param trackId stable track id.
     * @param feature torso Lab colour, or std::nullopt (ignored).
     */
    void observe(int trackId, const std::optional<cv::Vec3f>& feature);

    /// @return the voted team of a track (0, 1, kTeamOther) or std::nullopt if undecided.
    [[nodiscard]] std::optional<int> teamOf(int trackId) const;

    [[nodiscard]] bool ready() const { return ready_; }  ///< True once the two teams are known.
    /// @return drawing colour (BGR) of a team id (kTeamOther is grey).
    [[nodiscard]] cv::Vec3b color(int team) const;
    /// @return display name of a team id, e.g. "A (red)".
    [[nodiscard]] std::string name(int team) const;
    /// @return short label: "A", "B", "-" (other) or "?" (undecided/none).
    [[nodiscard]] std::string label(std::optional<int> team) const;

private:
    void fit();
    [[nodiscard]] int classify(const cv::Vec3f& feature) const;

    int warmupSamples_;
    int numTeams_ = 2;
    double threshold_ = 0.0;
    bool ready_ = false;
    std::vector<std::pair<int, cv::Vec3f>> pending_;          ///< Features collected before the fit.
    std::array<cv::Vec3f, 2> centers_{};                      ///< Lab cluster centres.
    std::unordered_map<int, std::array<float, 3>> votes_;     ///< Per-track votes {team0, team1, other}.
    std::array<cv::Vec3b, 2> colors_{};                       ///< Vivid BGR team colours.
    std::array<std::string, 2> names_{};                      ///< Team names.
};

}  // namespace ft
