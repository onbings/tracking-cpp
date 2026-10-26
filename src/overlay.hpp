/**
 * @file overlay.hpp
 * @brief Drawing helpers for the HUD, team panel, labels, trails, ball marker and sparklines.
 */
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ft::overlay {

inline const cv::Scalar kWhite{255, 255, 255};
inline const cv::Scalar kGrey{170, 170, 170};
inline const cv::Scalar kHudGreen{120, 255, 120};

/// @return a deterministic, well-spread BGR colour for a track id (golden-ratio hue walk).
cv::Vec3b colorForId(int id);

/// @brief Darkens a rectangle of the image in place (panel background).
void darkenRect(cv::Mat& img, cv::Rect rect, double alpha = 0.6);

/// @brief Draws a filled label with automatic black/white text, top-left at (x, y).
void drawLabel(cv::Mat& img, const std::string& text, int x, int y, const cv::Scalar& color, double scale = 0.45);

/// @brief Draws a dashed rectangle (used for predicted/implausible boxes).
void drawDashedRect(cv::Mat& img, cv::Point p1, cv::Point p2, const cv::Scalar& color, int dash = 8);

/// @brief Draws a fading polyline trail (thickness grows towards the present).
void drawTrail(cv::Mat& img, const std::vector<cv::Point2f>& points, const cv::Scalar& color);

/// @brief Draws a small line graph inside a box.
void drawSparkline(cv::Mat& img, const std::vector<double>& values, cv::Rect box,
                   const cv::Scalar& color = kHudGreen, const std::string& label = "");

/// @brief Draws a proportional bar of per-stage times (named segments).
void drawStageBar(cv::Mat& img, const std::vector<std::pair<std::string, double>>& stages, cv::Rect box);

/**
 * @brief Draws the HUD panel (top-left).
 * @param lines      (text, colour) rows.
 * @param fpsHistory values for the sparkline.
 * @param stageMs    (name, ms) pairs for the stage bar.
 * @param scale      text scale.
 */
void drawHud(cv::Mat& img, const std::vector<std::pair<std::string, cv::Scalar>>& lines,
             const std::vector<double>& fpsHistory, const std::vector<std::pair<std::string, double>>& stageMs,
             double scale);

/**
 * @brief Draws a text panel with an optional colour swatch per row.
 * @param rows       (text, textColour, swatch-or-none) rows; a swatch of (-1,-1,-1) means "no swatch".
 * @param topRight   top-right corner position.
 * @param scale      text scale.
 */
void drawPanel(cv::Mat& img, const std::vector<std::tuple<std::string, cv::Scalar, cv::Scalar>>& rows,
               cv::Point topRight, double scale);

/// @brief Draws the ball marker (circle + cross), trail and label.
void drawBall(cv::Mat& img, cv::Point center, int radius, const std::vector<cv::Point2f>& trail,
              const std::string& text, bool predicted);

/// @brief Draws the counting line with its IN/OUT text.
void drawCountingLine(cv::Mat& img, cv::Point2d a, cv::Point2d b, double scale, bool flash, const std::string& text);

/// @brief Draws a centred banner (used for the pause overlay).
void drawBanner(cv::Mat& img, const std::string& text);

}  // namespace ft::overlay
