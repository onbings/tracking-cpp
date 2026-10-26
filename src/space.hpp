/**
 * @file space.hpp
 * @brief Voronoi "space control" of the visible pitch and the top-down radar minimap.
 *
 * Each patch of visible turf belongs to the team of the closest player. Computing this on a metric
 * ground grid makes the shares true area ratios (a far patch counts as much as a near one). The same
 * raster drives the on-canvas overlay and the radar. See DOCUMENTATION.md §19.
 */
#pragma once

#include "geometry.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <optional>
#include <vector>

namespace ft {

/**
 * @brief Computes and renders team space control.
 */
class SpaceControl {
public:
    /// @param ground reference to the live ground model.
    explicit SpaceControl(const GroundModel& ground);

    /// @brief Sets the turf mask (binary, at @p scale x full resolution) restricting valid cells.
    void setTurf(const cv::Mat& grass, double scale);

    /**
     * @brief Computes the area share of each team over the visible turf.
     * @param positions player ground positions (m).
     * @param teams     team id of each player (same length).
     * @return team -> share in [0, 1], or std::nullopt if nothing is visible.
     */
    std::optional<std::map<int, double>> compute(const std::vector<cv::Point2d>& positions,
                                                 const std::vector<int>& teams);

    /// @return the last computed shares (empty before the first compute()).
    [[nodiscard]] const std::map<int, double>& shares() const { return shares_; }

    /**
     * @brief Tints the canvas turf with the controlling team's colour.
     * @param canvas    the display image (modified in place).
     * @param positions player ground positions (m).
     * @param teams     team ids.
     * @param colors    team -> BGR colour.
     */
    void overlay(cv::Mat& canvas, const std::vector<cv::Point2d>& positions, const std::vector<int>& teams,
                 const std::map<int, cv::Vec3b>& colors) const;

    /**
     * @brief Renders the top-down radar.
     * @param size    radar image size.
     * @param colors  team -> BGR colour.
     * @param players (ground position, BGR colour) of each player dot.
     * @param ball    ball ground position, if visible.
     * @return the radar image, or empty if there is no raster yet.
     */
    [[nodiscard]] cv::Mat minimap(cv::Size size, const std::map<int, cv::Vec3b>& colors,
                                  const std::vector<std::pair<cv::Point2d, cv::Vec3b>>& players,
                                  const std::optional<cv::Point2d>& ball) const;

private:
    const GroundModel& ground_;
    double step_ = 0.5;          ///< Grid spacing (m).
    std::size_t maxCells_ = 30000;
    cv::Mat raster_;             ///< Team id per grid cell (int16), -2 = outside turf.
    cv::Vec4d bounds_{};         ///< Metric bounds of the raster.
    bool hasRaster_ = false;
    std::map<int, double> shares_;
    cv::Mat turf_;              ///< Closed turf mask.
    double turfScale_ = 1.0;

    [[nodiscard]] bool onTurf(const cv::Point2d& imagePt) const;
};

}  // namespace ft
