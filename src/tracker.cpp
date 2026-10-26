/**
 * @file tracker.cpp
 * @brief Implementation of the BoT-SORT-style tracker (BYTE association + greedy IoU matching).
 */
#include "tracker.hpp"

#include <algorithm>
#include <limits>

namespace ft {

BotSortTracker::BotSortTracker(float highThresh, int maxAge, int nInit)
    : highThresh_(highThresh), maxAge_(maxAge), nInit_(nInit) {}

namespace {
/**
 * @brief Greedy IoU matching: repeatedly link the highest-IoU track/detection pair above a threshold.
 * @param trackBoxes      predicted track boxes.
 * @param detBoxes        candidate detection boxes.
 * @param trackIdx,detIdx indices (into the tracker's arrays) of each entry.
 * @param iouThr          minimum IoU to accept a pair.
 * @param[out] matches    pairs (trackIdx, detIdx).
 * @param[out] freeDet    detection indices left unmatched.
 */
void greedyMatch(const std::vector<cv::Rect2f>& trackBoxes, const std::vector<cv::Rect2f>& detBoxes,
                 const std::vector<int>& trackIdx, const std::vector<int>& detIdx, double iouThr,
                 std::vector<std::pair<int, int>>& matches, std::vector<int>& freeTrack, std::vector<int>& freeDet) {
    std::vector<char> tUsed(trackBoxes.size(), 0), dUsed(detBoxes.size(), 0);
    while (true) {
        double best = iouThr;
        int bt = -1, bd = -1;
        for (std::size_t t = 0; t < trackBoxes.size(); ++t) {
            if (tUsed[t]) continue;
            for (std::size_t d = 0; d < detBoxes.size(); ++d) {
                if (dUsed[d]) continue;
                const double v = iou(trackBoxes[t], detBoxes[d]);
                if (v > best) { best = v; bt = static_cast<int>(t); bd = static_cast<int>(d); }
            }
        }
        if (bt < 0) break;
        tUsed[bt] = dUsed[bd] = 1;
        matches.emplace_back(trackIdx[bt], detIdx[bd]);
    }
    for (std::size_t t = 0; t < trackBoxes.size(); ++t) if (!tUsed[t]) freeTrack.push_back(trackIdx[t]);
    for (std::size_t d = 0; d < detBoxes.size(); ++d) if (!dUsed[d]) freeDet.push_back(detIdx[d]);
}
}  // namespace

std::vector<Track> BotSortTracker::update(const std::vector<Detection>& detections, const cv::Matx33d& warp) {
    // 1. Predict every track and compensate for camera motion.
    for (auto& t : tracks_) {
        t.kf.predict();
        t.kf.applyWarp(warp);
        t.age += 1;
    }

    // 2. Split detections into high- and low-confidence sets (BYTE).
    std::vector<int> highDet, lowDet;
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        const float s = detections[i].score;
        if (s >= highThresh_) highDet.push_back(i);
        else if (s >= lowThresh_) lowDet.push_back(i);
    }

    std::vector<cv::Rect2f> trackBoxes;
    std::vector<int> trackIdx;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        trackBoxes.push_back(tracks_[i].kf.box());
        trackIdx.push_back(i);
    }

    auto boxesOf = [&](const std::vector<int>& idx) {
        std::vector<cv::Rect2f> b;
        for (int i : idx) b.push_back(detections[i].box);
        return b;
    };
    auto boxesOfTracks = [&](const std::vector<int>& idx) {
        std::vector<cv::Rect2f> b;
        for (int i : idx) b.push_back(tracks_[i].kf.box());
        return b;
    };

    // Confirmed tracks are matched before tentative ones, so a new track cannot steal a detection
    // from an established identity (reduces ID switches, like ultralytics' association order).
    std::vector<int> confirmedIdx, tentativeIdx;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        (tracks_[i].confirmed ? confirmedIdx : tentativeIdx).push_back(i);
    }

    // 3a. First association: confirmed tracks vs high-confidence detections.
    std::vector<std::pair<int, int>> matches;
    std::vector<int> freeConfirmed, freeDetHigh;
    greedyMatch(boxesOfTracks(confirmedIdx), boxesOf(highDet), confirmedIdx, highDet, 0.2, matches, freeConfirmed, freeDetHigh);

    // 3b. Second association: still-unmatched confirmed tracks vs low-confidence detections (occlusions).
    std::vector<std::pair<int, int>> matches2;
    std::vector<int> stillFreeConfirmed, unusedLow;
    greedyMatch(boxesOfTracks(freeConfirmed), boxesOf(lowDet), freeConfirmed, lowDet, 0.3, matches2, stillFreeConfirmed, unusedLow);
    matches.insert(matches.end(), matches2.begin(), matches2.end());

    // 3c. Tentative (unconfirmed) tracks compete only for the leftover high-confidence detections.
    std::vector<std::pair<int, int>> matches3;
    std::vector<int> freeTentative, freeDetFinal;
    greedyMatch(boxesOfTracks(tentativeIdx), boxesOf(freeDetHigh), tentativeIdx, freeDetHigh, 0.2, matches3, freeTentative, freeDetFinal);
    matches.insert(matches.end(), matches3.begin(), matches3.end());

    // 4. Apply matches.
    for (auto [ti, di] : matches) {
        auto& t = tracks_[ti];
        t.kf.update(detections[di].box);
        t.score = detections[di].score;
        t.classId = detections[di].classId;
        t.hits += 1;
        t.age = 0;
        if (t.hits >= nInit_) t.confirmed = true;
    }

    // 5. Start new tracks from the high-confidence detections that nothing matched.
    for (int di : freeDetFinal) {
        const auto& d = detections[di];
        if (d.score < highThresh_) continue;
        tracks_.push_back({nextId_++, KalmanBox(d.box), d.classId, d.score});
    }

    // 6. Drop stale tracks.
    std::erase_if(tracks_, [this](const Trk& t) { return t.age > maxAge_; });

    // 7. Report confirmed, recently-seen tracks.
    std::vector<Track> out;
    for (const auto& t : tracks_) {
        if (t.confirmed && t.age == 0) out.push_back({t.id, t.kf.box(), t.score, t.classId});
    }
    return out;
}

}  // namespace ft
