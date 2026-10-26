/**
 * @file app.cpp
 * @brief Implementation of the TrackingApp orchestration, rendering and reports.
 *
 * Mirrors the Python TrackingApp: a decoder thread feeds a detector thread, which feeds the main
 * loop (camera motion -> ground model -> tracker -> teams -> kinematics -> ball -> space -> render).
 * Detection runs one frame ahead so GPU and CPU work overlap and the laptop GPU keeps its clocks.
 */
#include "app.hpp"

#include "analytics.hpp"
#include "ball.hpp"
#include "bounded_queue.hpp"
#include "camera_motion.hpp"
#include "common.hpp"
#include "detector.hpp"
#include "geometry.hpp"
#include "kinematics.hpp"
#include "log.hpp"
#include "overlay.hpp"
#include "perf.hpp"
#include "runtime_env.hpp"
#include "space.hpp"
#include "team.hpp"
#include "tracker.hpp"
#include "video_reader.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace ft {

using json = nlohmann::json;
namespace {

constexpr int kSpaceEvery = 3;             // frames between space-control updates
constexpr double kPossessionRadius = 1.5;  // metres
constexpr int kMinGoodWidth = 1280;        // below this, players/ball get too small for reliable detection
const char* kWindow = "Football tracker";

/// @brief Formats seconds as m:ss.s or h:mm:ss.s.
std::string clockStr(double seconds) {
    seconds = std::max(0.0, seconds);
    const int h = static_cast<int>(seconds) / 3600;
    const int m = (static_cast<int>(seconds) / 60) % 60;
    const double s = std::fmod(seconds, 60.0);
    char buf[32];
    if (h > 0) std::snprintf(buf, sizeof(buf), "%d:%02d:%04.1f", h, m, s);
    else std::snprintf(buf, sizeof(buf), "%02d:%04.1f", m, s);
    return buf;
}

/// @brief A detection result passed from the detector thread to the main loop.
struct DetPacket {
    FramePacket frame;
    std::vector<Detection> detections;
    double detectMs = 0.0;
};
}  // namespace

// ---------------------------------------------------------------------------- Impl

struct TrackingApp::Impl {
    explicit Impl(Config cfg);

    int run();

    Config cfg;
    std::string device;
    bool useCuda = false;

    std::unique_ptr<YoloDetector> detector;
    std::optional<std::vector<int>> trackClasses;   // std::nullopt = all
    std::vector<int> detectClasses;                 // persons (+ ball)
    std::unique_ptr<VideoReader> reader;
    BotSortTracker tracker;
    CameraMotion* camera = nullptr;
    std::unique_ptr<CameraMotion> cameraOwned;
    std::unique_ptr<GroundModel> ground;
    std::unique_ptr<Kinematics> kin;
    std::unique_ptr<Kinematics> ballKin;
    std::unique_ptr<BallTracker> ball;
    std::unique_ptr<TeamClassifier> teams;
    std::unique_ptr<SpaceControl> space;
    std::unique_ptr<Trails> trails;
    std::unique_ptr<Heatmap> heatmap;
    std::unique_ptr<LineCounter> line;
    PerfMonitor perf;
    SystemMonitor sysmon;

    double scale = 1.0;
    cv::Size canvasSize;
    double hudScale = 0.5;

    // UI toggles
    bool showHud = true, showPanel = true, showRadar = true, showVoronoi = false;
    bool showTrails = true, showHeatmap = false, showLine = true;
    bool paused = false, stepOnce = false, running = true;

    // per-frame working state
    struct PlayerInfo { int id; std::optional<int> team; const MotionState* state; };
    std::vector<PlayerInfo> framePlayers;
    int lastDets = 0, activeTracks = 0, lastIndex = -1;
    cv::Mat lastCanvas;

    // accumulated football stats
    std::map<int, double> controlSum;
    int controlSamples = 0;
    std::map<int, long> possession;
    int ballFrames = 0, ballRoiHits = 0;

    // input-robustness checks (other clips: broadcast cuts, close-ups, broken metadata)
    int shotCuts = 0, nonPitchFrames = 0, lastCutFrame = -1000000;
    double turfSum = 0.0;

    void onShotCut(int idx);
    [[nodiscard]] std::vector<std::string> sourceWarnings() const;

    // outputs
    cv::VideoWriter writer;
    std::ofstream csv;

    void process(const DetPacket& packet);
    void splitDetections(const std::vector<Detection>& dets, std::vector<Detection>& players, std::vector<Detection>& balls) const;
    void updatePlayers(const std::vector<Track>& tracks, const cv::Mat& frame, const cv::Matx33d& H, int idx);
    void updateBall(const std::vector<Detection>& balls, const cv::Mat& frame, const cv::Matx33d& H, int idx);
    void annotate(cv::Mat& canvas, const std::vector<Track>& tracks, int idx, const cv::Matx33d& H);
    void drawHud(cv::Mat& canvas, int idx);
    void drawPanel(cv::Mat& canvas);
    void drawRadar(cv::Mat& canvas);
    cv::Vec3b teamColor(int id, std::optional<int> team) const;
    void handleKey(int key);
    bool windowClosed() const;
    void openOutputs();
    void report();
    json footballStats(double minSeconds = 1.0) const;
};

TrackingApp::Impl::Impl(Config c)
    : cfg(std::move(c)), tracker(cfg.conf, cfg.maxAge) {
    // Resolve the device: try CUDA, fall back to CPU on "auto".
    device = cfg.device;
    useCuda = device == "auto" || device.rfind("cuda", 0) == 0;

    // Low detector threshold so BYTE's second stage and the ball get their low-score boxes.
    const float detConf = cfg.ball ? std::min(cfg.conf, 0.1f) : cfg.conf;
    detector = std::make_unique<YoloDetector>(cfg.modelsDir, cfg.model, cfg.device, cfg.half, cfg.imgSize, detConf, cfg.iou);
    useCuda = detector->provider() == "CUDA";

    trackClasses = parseClasses(cfg.classes);
    if (!trackClasses) {
        detectClasses.clear();  // all
    } else {
        detectClasses = *trackClasses;
        if (cfg.ball && std::ranges::find(detectClasses, kBallClass) == detectClasses.end()) {
            detectClasses.push_back(kBallClass);
        }
    }
    log::info("Detector: {} {} @{}, provider {}, ball={}", cfg.model, detector->fp16() ? "FP16" : "FP32",
              cfg.imgSize, detector->provider(), cfg.ball);

    reader = std::make_unique<VideoReader>(cfg.source, cfg.queueSize, cfg.startFrame);
    if (cfg.fps > 0) reader->setFps(cfg.fps);
    log::info("Source: {} | {}x{} @ {:.2f} fps ({}) | {} frames | {}{} | start={}", cfg.source, reader->width(),
              reader->height(), reader->fps(), reader->fpsSource(), reader->frameCount(), reader->backend(),
              reader->hardwareAccelerated() ? " (HW)" : "", reader->startFrame());
    if (reader->width() < kMinGoodWidth) {
        log::warn("Low resolution ({}x{}): distant players and the ball cover few pixels, expect missed detections.",
                  reader->width(), reader->height());
    }
    log::info("Metric scale assumes --hfov {:.0f} deg and --person-height {:.2f} m: set them for another camera.",
              cfg.hfovDeg, cfg.personHeight);

    scale = std::min(1.0, cfg.displayWidth / static_cast<double>(reader->width()));
    canvasSize = {static_cast<int>(std::lround(reader->width() * scale)),
                  static_cast<int>(std::lround(reader->height() * scale))};
    hudScale = std::clamp(canvasSize.width / 2600.0, 0.4, 0.7);

    cameraOwned = std::make_unique<CameraMotion>(cv::Size(reader->width(), reader->height()));
    camera = cameraOwned.get();
    ground = std::make_unique<GroundModel>(reader->width(), reader->height(), cfg.hfovDeg, cfg.personHeight, cfg.maxDepth);
    kin = std::make_unique<Kinematics>(reader->fps(), 4.0, 40.0, 15, 1.0, 0.5);
    ballKin = std::make_unique<Kinematics>(reader->fps(), 15.0, 90.0, static_cast<int>(reader->fps() * 0.8), 1.5, 0.15);
    ball = std::make_unique<BallTracker>(reader->fps());
    if (cfg.teams) teams = std::make_unique<TeamClassifier>();
    space = std::make_unique<SpaceControl>(*ground);
    trails = std::make_unique<Trails>(static_cast<std::size_t>(cfg.trailLength));
    heatmap = std::make_unique<Heatmap>(canvasSize);
    if (cfg.line) {
        const auto& l = *cfg.line;
        line = std::make_unique<LineCounter>(cv::Point2d(l[0] * reader->width(), l[1] * reader->height()),
                                             cv::Point2d(l[2] * reader->width(), l[3] * reader->height()));
    }
    showVoronoi = cfg.voronoi;
}

void TrackingApp::Impl::splitDetections(const std::vector<Detection>& dets, std::vector<Detection>& players,
                                        std::vector<Detection>& balls) const {
    for (const auto& d : dets) {
        const bool isBall = d.classId == kBallClass;
        if (cfg.ball && isBall) { balls.push_back(d); continue; }
        if (!trackClasses || std::ranges::find(*trackClasses, d.classId) != trackClasses->end()) {
            if (!(isBall && cfg.ball)) players.push_back(d);
        }
    }
}

cv::Vec3b TrackingApp::Impl::teamColor(int id, std::optional<int> team) const {
    if (!teams || !teams->ready()) return overlay::colorForId(id);
    if (!team) return {160, 160, 160};
    return teams->color(*team);
}

void TrackingApp::Impl::process(const DetPacket& packet) {
    const int idx = packet.frame.index;
    const cv::Mat& frame = packet.frame.frame;
    perf.record("decode", packet.frame.decodeMs);
    perf.record("detect", packet.detectMs);

    std::vector<Detection> players, balls;
    splitDetections(packet.detections, players, balls);
    std::vector<Detection> strong;
    for (const auto& d : players) if (d.score >= cfg.conf) strong.push_back(d);
    lastDets = static_cast<int>(strong.size());

    cv::Mat canvas;
    cv::Matx33d H;
    {
        auto t = perf.measure("camera");
        cv::resize(frame, canvas, canvasSize, 0, 0, cv::INTER_LINEAR);
        std::vector<cv::Rect2f> boxes;
        for (const auto& d : strong) boxes.push_back(d.box);
        H = camera->update(canvas, boxes);
        turfSum += camera->lowerTurfRatio();
        if (camera->cut()) onShotCut(idx);
        if (!camera->pitchView()) ++nonPitchFrames;
        std::vector<cv::Rect2f> people;
        for (const auto& d : strong) if (d.classId == kPersonClass) people.push_back(d.box);
        std::vector<cv::Rect2f> plausible;
        // Close-ups and crowd shots would feed the calibration with people standing off the pitch plane.
        if (camera->pitchView()) {
            for (const auto& b : people) if (ground->plausiblePerson(b)) plausible.push_back(b);
        }
        ground->addObservations(plausible);
        ground->update(idx);
        if (!camera->grass().empty()) space->setTurf(camera->grass(), camera->scale());
    }

    std::vector<Track> tracks;
    {
        auto t = perf.measure("track");
        tracks = tracker.update(players, H);
    }
    {
        auto t = perf.measure("analytics");
        updatePlayers(tracks, frame, H, idx);
    }
    {
        auto t = perf.measure("ball");
        updateBall(balls, frame, H, idx);
    }
    {
        auto t = perf.measure("render");
        annotate(canvas, tracks, idx, H);
        if (writer.isOpened()) writer.write(canvas);
        if (cfg.show) cv::imshow(kWindow, canvas);
    }
    lastCanvas = canvas;
    lastIndex = idx;
}

void TrackingApp::Impl::onShotCut(int idx) {
    // A new shot may come from another camera at another place: nothing measured before is comparable.
    ++shotCuts;
    lastCutFrame = idx;
    tracker.reset();          // old boxes would be matched to unrelated players
    ground->reset();          // recalibrate tilt/height for the new view
    kin->invalidate();        // no displacement across the cut
    ballKin->reset("ball");
    ball->reset();
    log::info("Shot cut at frame {} (histogram correlation {:.2f}): tracks and calibration reset", idx,
              camera->histCorrelation());
}

std::vector<std::string> TrackingApp::Impl::sourceWarnings() const {
    std::vector<std::string> w;
    const int frames = std::max(1, perf.frames());
    if (reader->fpsSource() == "fallback") {
        w.emplace_back("Frame rate missing/implausible in the file: 30 fps assumed. Speeds scale with it; use --fps.");
    }
    if (shotCuts > 0) {
        w.push_back(std::to_string(shotCuts) + " shot cut(s) detected: tracks and calibration were reset at each one, "
                    "so identities and per-player totals are split per shot (edited/broadcast footage).");
    }
    if (nonPitchFrames > frames / 10) {
        w.push_back(std::to_string(nonPitchFrames * 100 / frames) + "% of frames show little turf (close-ups, crowd, "
                    "replays): ground analytics were paused there.");
    }
    if (reader->width() < kMinGoodWidth) {
        w.push_back("Low source resolution (" + std::to_string(reader->width()) + "px wide): expect missed players/ball.");
    }
    if (!ground->ready()) {
        w.emplace_back("Camera never calibrated: no metric output. Needs a wide elevated view with fully visible players.");
    } else if (ground->residual() > 0.25) {
        w.emplace_back("Poor camera fit (residual > 25%): the view may not match the elevated pinhole model; "
                       "check --hfov/--person-height.");
    }
    if (teams && !teams->ready()) w.emplace_back("Teams never separated: the two kits may be too similar (try --no-teams).");
    return w;
}

void TrackingApp::Impl::updatePlayers(const std::vector<Track>& tracks, const cv::Mat& frame, const cv::Matx33d& H, int idx) {
    const bool measurable = camera->ok() && camera->pitchView();
    if (measurable) kin->warp(H); else kin->invalidate();
    framePlayers.clear();

    for (const auto& trk : tracks) {
        if (!ground->plausiblePerson(trk.box)) continue;
        const cv::Point2d foot = trk.foot();
        if (teams && (idx % 2 == 0 || !teams->ready())) teams->observe(trk.id, jerseyFeature(frame, trk.box));
        const double pixelSigma = std::max(2.0, 0.04 * trk.box.height);
        const MotionState& s = kin->observe(std::to_string(trk.id), foot, *ground, idx, pixelSigma);
        std::optional<int> team = teams ? teams->teamOf(trk.id) : std::nullopt;
        framePlayers.push_back({trk.id, team, &s});
    }
    activeTracks = static_cast<int>(framePlayers.size());
    const auto stale = kin->prune(idx, cfg.maxAge * 3);
    if (line) {
        std::vector<int> ids;
        for (const auto& k : stale) ids.push_back(std::stoi(k));
        line->forget(ids);
    }

    if (teams && teams->ready() && camera->pitchView() && idx % kSpaceEvery == 0) {
        std::vector<cv::Point2d> pos;
        std::vector<int> tms;
        for (const auto& p : framePlayers) {
            if (p.team && *p.team != kTeamOther && p.state->hasGround) {
                pos.push_back(p.state->ground);
                tms.push_back(*p.team);
            }
        }
        if (!pos.empty()) {
            auto shares = space->compute(pos, tms);
            if (shares && shares->size() >= 2) {
                ++controlSamples;
                for (auto [team, v] : *shares) controlSum[team] += v;
            }
        }
    }
}

void TrackingApp::Impl::updateBall(const std::vector<Detection>& balls, const cv::Mat& frame, const cv::Matx33d& H, int idx) {
    if (!cfg.ball) return;
    ball->warp(H);
    if (camera->ok() && camera->pitchView()) ballKin->warp(H); else ballKin->invalidate();

    auto cands = plausibleBallSize(ballCandidates(balls, frame), *ground);
    if (cfg.ballRoi && cands.empty() && ball->active()) {
        if (auto roi = ball->roi(frame.size())) {
            cv::Mat crop = frame(*roi).clone();
            std::vector<int> ballOnly{kBallClass};
            auto rd = detector->detect(crop, ballOnly, 640);
            for (auto& d : rd) { d.box.x += roi->x; d.box.y += roi->y; }
            cands = plausibleBallSize(ballCandidates(rd, frame), *ground);
            if (!cands.empty()) ++ballRoiHits;
        }
    }

    if (!ball->update(cands)) return;
    ++ballFrames;
    if (ball->reacquired()) ballKin->reset("ball");
    const auto gp = ball->groundPoint();
    if (!gp) return;
    const double pixelSigma = std::max(2.0, 0.25 * ball->size());
    const MotionState& bs = ballKin->observe("ball", *gp, *ground, idx, pixelSigma);
    if (!bs.hasGround || !teams) return;

    std::optional<int> best;
    double bestD = kPossessionRadius;
    for (const auto& p : framePlayers) {
        if (!p.team || *p.team == kTeamOther || !p.state->hasGround) continue;
        const double d = cv::norm(p.state->ground - bs.ground);
        if (d < bestD) { bestD = d; best = p.team; }
    }
    if (best) ++possession[*best];
}

void TrackingApp::Impl::annotate(cv::Mat& canvas, const std::vector<Track>& tracks, int idx, const cv::Matx33d& H) {
    const double s = scale;
    const double fps = reader->fps();
    const double tVideo = idx / fps;

    // State lookup for this frame.
    std::unordered_map<int, PlayerInfo> byId;
    std::vector<int> aliveIds;
    for (const auto& p : framePlayers) { byId[p.id] = p; aliveIds.push_back(p.id); }

    // Warp trails and heatmap into the current canvas frame.
    const cv::Matx33d scaleM(s, 0, 0, 0, s, 0, 0, 0, 1);
    const cv::Matx33d scaleInv(1 / s, 0, 0, 0, 1 / s, 0, 0, 0, 1);
    const cv::Matx33d canvasH = scaleM * H * scaleInv;
    trails->warp(canvasH);
    heatmap->warp(canvasH);

    // Voronoi overlay.
    if (showVoronoi && teams && teams->ready()) {
        std::vector<cv::Point2d> pos;
        std::vector<int> tms;
        for (const auto& p : framePlayers) {
            if (p.team && *p.team != kTeamOther && p.state->hasGround) { pos.push_back(p.state->ground); tms.push_back(*p.team); }
        }
        if (!pos.empty()) {
            std::map<int, cv::Vec3b> colors;
            for (int t = 0; t < 2; ++t) colors[t] = teams->color(t);
            space->overlay(canvas, pos, tms, colors);
        }
    }

    std::vector<cv::Point2f> heatPoints;
    for (const auto& trk : tracks) {
        const PlayerInfo* info = byId.count(trk.id) ? &byId.at(trk.id) : nullptr;
        const cv::Point p1(static_cast<int>(trk.box.x * s), static_cast<int>(trk.box.y * s));
        const cv::Point p2(static_cast<int>((trk.box.x + trk.box.width) * s),
                           static_cast<int>((trk.box.y + trk.box.height) * s));
        const std::optional<int> team = info ? info->team : std::nullopt;
        const cv::Vec3b bgr = teamColor(trk.id, team);
        const cv::Scalar color(bgr[0], bgr[1], bgr[2]);

        if (!info) {  // implausible (e.g. banner) or just not analysed this frame
            overlay::drawDashedRect(canvas, p1, p2, color);
            continue;
        }
        const cv::Point2d foot = trk.foot();
        const cv::Point2f footCanvas(static_cast<float>(foot.x * s), static_cast<float>(foot.y * s));
        trails->add(trk.id, footCanvas);
        heatPoints.push_back(footCanvas);
        if (line) line->update(trk.id, foot, idx);
        if (showTrails) {
            std::vector<cv::Point2f> pts(trails->get(trk.id).begin(), trails->get(trk.id).end());
            overlay::drawTrail(canvas, pts, color);
        }
        const int rx = std::max(6, (p2.x - p1.x) / 2 + 4);
        cv::ellipse(canvas, {static_cast<int>(footCanvas.x), static_cast<int>(footCanvas.y)},
                    {rx, std::max(3, rx / 3)}, 0, -45, 235, color, 2, cv::LINE_AA);
        cv::rectangle(canvas, p1, p2, color, 1);
        const double speed = info->state->speedKmh();
        std::string label = "#" + std::to_string(trk.id);
        if (teams) label += " " + teams->label(team);
        if (info->state->hasFilter) label += " " + std::to_string(static_cast<int>(speed)) + "km/h " +
                                             std::to_string(static_cast<int>(info->state->distance)) + "m";
        overlay::drawLabel(canvas, label, p1.x, p1.y, color, 0.42);

        if (csv.is_open()) {
            const auto& st = *info->state;
            csv << idx << ',' << tVideo << ',' << trk.id << ',' << cocoNames()[trk.classId] << ','
                << (teams ? teams->label(team) : "") << ',' << trk.score << ','
                << trk.box.x << ',' << trk.box.y << ',' << (trk.box.x + trk.box.width) << ',' << (trk.box.y + trk.box.height) << ','
                << (st.hasGround ? st.ground.x : 0.0) << ',' << (st.hasGround ? st.ground.y : 0.0) << ','
                << (st.hasFilter ? speed : 0.0) << ',' << st.distance << '\n';
        }
    }
    trails->keepOnly(aliveIds);
    heatmap->add(heatPoints);
    if (showHeatmap) heatmap->render(canvas);

    if (line && showLine) {
        const bool flash = idx - line->lastEventFrame() < 8;
        const std::string text = "IN " + std::to_string(line->countIn()) + "  OUT " + std::to_string(line->countOut());
        overlay::drawCountingLine(canvas, line->a(), line->b(), s, flash, text);
    }

    if (cfg.ball && ball->active()) {
        const cv::Point c(static_cast<int>(ball->position().x * s), static_cast<int>(ball->position().y * s));
        const MotionState* bs = ballKin->find("ball");
        std::string txt = "ball";
        if (bs && bs->hasFilter) txt = "ball " + std::to_string(static_cast<int>(bs->speedKmh())) + "km/h";
        std::vector<cv::Point2f> trail;
        for (const auto& p : ball->trail()) trail.emplace_back(static_cast<float>(p.x * s), static_cast<float>(p.y * s));
        overlay::drawBall(canvas, c, static_cast<int>(ball->size() * s / 2), trail, txt, !ball->visible());
    }

    if (showHud) drawHud(canvas, idx);
    if (showPanel && (teams || cfg.ball)) drawPanel(canvas);
    if (showRadar) drawRadar(canvas);
}

void TrackingApp::Impl::drawHud(cv::Mat& canvas, int idx) {
    const auto snap = sysmon.snapshot();
    const int total = reader->frameCount();
    const double fps = perf.fps();
    const std::string progress = total ? std::to_string(idx + 1) + "/" + std::to_string(total) : std::to_string(idx + 1);
    const std::string eta = (total && fps > 0) ? " ETA " + clockStr((total - idx - 1) / fps) : "";
    const double realtime = reader->fps() > 0 ? fps / reader->fps() : 0;

    std::vector<std::pair<std::string, cv::Scalar>> lines;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s@%d | %s %s | BoT-SORT+CMC | %dx%d", cfg.model.c_str(), cfg.imgSize,
                  detector->provider().c_str(), detector->fp16() ? "FP16" : "FP32", reader->width(), reader->height());
    lines.emplace_back(buf, overlay::kWhite);
    std::snprintf(buf, sizeof(buf), "FPS %5.1f (avg %4.1f, %.2fx) | frame %s | t=%s%s", fps, perf.avgFps(), realtime,
                  progress.c_str(), clockStr(idx / reader->fps()).c_str(), eta.c_str());
    lines.emplace_back(buf, overlay::kHudGreen);
    std::string ms = "ms ";
    for (const auto& st : {"decode", "detect", "camera", "track", "analytics", "ball", "render"}) {
        ms += std::string(st).substr(0, 4) + " " + std::to_string(static_cast<int>(perf.mean(st))) + " ";
    }
    ms += "| q " + std::to_string(reader->queueDepth()) + "/" + std::to_string(reader->queueCapacity()) +
          " | " + perf.bottleneck();
    lines.emplace_back(ms, overlay::kWhite);
    {
        // Source health: red while a cut was just handled or the view shows too little pitch.
        const bool recentCut = idx - lastCutFrame < static_cast<int>(reader->fps());
        std::snprintf(buf, sizeof(buf), "src %.0f fps (%s) | pitch turf %.0f%% | cuts %d", reader->fps(),
                      reader->fpsSource().c_str(), camera->lowerTurfRatio() * 100, shotCuts);
        std::string src = buf;
        if (recentCut) src += " | SHOT CUT: reset";
        else if (!camera->pitchView()) src += " | NON-PITCH VIEW: analytics paused";
        lines.emplace_back(src, recentCut || !camera->pitchView() ? cv::Scalar(60, 60, 255) : overlay::kGrey);
    }
    if (snap.rssMb > 0) {
        std::snprintf(buf, sizeof(buf), "CPU %3.0f%% | RSS %5.0f MB", snap.cpuPercent, snap.rssMb);
        lines.emplace_back(buf, overlay::kGrey);
    }
    if (snap.hasGpu) {
        std::snprintf(buf, sizeof(buf), "GPU %3.0f%% | VRAM %.0f/%.0f MB | %.0fC %.1fW", snap.gpuUtil, snap.gpuMemMb,
                      snap.gpuMemTotalMb, snap.gpuTempC, snap.gpuPowerW);
        lines.emplace_back(buf, overlay::kGrey);
    }
    std::snprintf(buf, sizeof(buf), "dets %d | active %d", lastDets, activeTracks);
    lines.emplace_back(buf, cv::Scalar(120, 220, 255));
    lines.emplace_back("[q]uit [space]pause [n]ext [h]ud [p]anel [r]adar [v]oronoi [t]rails [m]ap [l]ine [s]nap",
                       overlay::kGrey);

    std::vector<std::pair<std::string, double>> stageMs;
    for (const auto& st : PerfMonitor::kPipeline) stageMs.emplace_back(st, perf.mean(st));
    std::vector<double> fpsHist(perf.fpsHistory().begin(), perf.fpsHistory().end());
    overlay::drawHud(canvas, lines, fpsHist, stageMs, hudScale);
}

void TrackingApp::Impl::drawPanel(cv::Mat& canvas) {
    std::vector<std::tuple<std::string, cv::Scalar, cv::Scalar>> rows;
    const cv::Scalar noSwatch(-1, -1, -1);
    if (teams && teams->ready()) {
        std::map<int, int> visible;
        for (const auto& p : framePlayers) if (p.team) visible[*p.team]++;
        long possTotal = 0;
        for (auto [t, n] : possession) possTotal += n;
        std::map<int, double> dist;
        std::map<int, double> topSpeed;
        std::map<int, int> topId;
        for (const auto& [key, st] : kin->all()) {
            const auto team = teams->teamOf(std::stoi(key));
            if (!team || *team == kTeamOther) continue;
            dist[*team] += st.distance;
            if (st.maxSpeedKmh > topSpeed[*team]) { topSpeed[*team] = st.maxSpeedKmh; topId[*team] = std::stoi(key); }
        }
        for (int t = 0; t < 2; ++t) {
            char buf[256];
            const double ctrl = space->shares().count(t) ? space->shares().at(t) : -1;
            const double avg = controlSamples ? controlSum[t] / controlSamples : -1;
            const double poss = possTotal ? static_cast<double>(possession[t]) / possTotal : -1;
            std::string txt = teams->name(t) + ": " + std::to_string(visible[t]) + " on screen | space ";
            txt += ctrl >= 0 ? std::to_string(static_cast<int>(ctrl * 100)) + "%" : "--";
            if (avg >= 0) txt += " (avg " + std::to_string(static_cast<int>(avg * 100)) + "%)";
            if (poss >= 0) txt += " | poss " + std::to_string(static_cast<int>(poss * 100)) + "%";
            std::snprintf(buf, sizeof(buf), " | dist %.0f m", dist[t]);
            txt += buf;
            if (topSpeed[t] > 0) {
                std::snprintf(buf, sizeof(buf), " | top #%d %.1f km/h", topId[t], topSpeed[t]);
                txt += buf;
            }
            const cv::Vec3b c = teams->color(t);
            rows.emplace_back(txt, overlay::kWhite, cv::Scalar(c[0], c[1], c[2]));
        }
    } else if (teams) {
        rows.emplace_back("Teams: learning jersey colours...", overlay::kGrey, noSwatch);
    }
    if (cfg.ball) {
        const MotionState* b = ballKin->find("ball");
        const double seen = static_cast<double>(ballFrames) / std::max(1, perf.frames());
        char buf[256];
        if (b) {
            std::snprintf(buf, sizeof(buf), "Ball: %.0f km/h | max %.0f km/h | %.0f m | detected %.0f%% of frames",
                          b->hasFilter ? b->speedKmh() : 0.0, b->maxSpeedKmh, b->distance, seen * 100);
        } else {
            std::snprintf(buf, sizeof(buf), "Ball: searching... | detected %.0f%% of frames", seen * 100);
        }
        rows.emplace_back(buf, cv::Scalar(0, 220, 255), cv::Scalar(255, 255, 255));
    }
    char buf[256];
    if (ground->ready()) {
        std::snprintf(buf, sizeof(buf), "Camera: tilt %.0f deg, height %.1f m, fit err %.0f%% | motion %d inliers",
                      ground->tiltDeg(), ground->camHeight(), ground->residual() * 100, camera->inliers());
    } else {
        std::snprintf(buf, sizeof(buf), "Camera: calibrating from player heights...");
    }
    rows.emplace_back(buf, overlay::kGrey, noSwatch);
    overlay::drawPanel(canvas, rows, {canvas.cols - 10, 10}, hudScale * 0.95);
}

void TrackingApp::Impl::drawRadar(cv::Mat& canvas) {
    if (!ground->ready()) return;
    std::map<int, cv::Vec3b> colors;
    if (teams && teams->ready()) for (int t = 0; t < 2; ++t) colors[t] = teams->color(t);
    std::vector<std::pair<cv::Point2d, cv::Vec3b>> players;
    for (const auto& p : framePlayers) {
        if (p.state->hasGround) players.emplace_back(p.state->ground, teamColor(p.id, p.team));
    }
    if (space->shares().empty() || !space->minimap({1, 1}, {}, {}, {}).empty()) {
        // ensure we have a raster even when space wasn't computed this frame
    }
    std::optional<cv::Point2d> ballPos;
    const MotionState* bs = ballKin->find("ball");
    if (bs && ball->visible()) ballPos = bs->ground;
    const int w = static_cast<int>(canvasSize.width * 0.22);
    cv::Mat radar = space->minimap({w, static_cast<int>(w * 0.75)}, colors, players, ballPos);
    if (radar.empty()) return;
    const int y0 = canvas.rows - radar.rows - 10, x0 = canvas.cols - radar.cols - 10;
    cv::Mat roi = canvas(cv::Rect(x0, y0, radar.cols, radar.rows));
    cv::addWeighted(roi, 0.15, radar, 0.85, 0, roi);
}

void TrackingApp::Impl::handleKey(int key) {
    switch (key) {
        case 'q': case 27: running = false; break;
        case ' ':
            paused = !paused;
            if (paused && !lastCanvas.empty()) {
                cv::Mat p = lastCanvas.clone();
                overlay::drawBanner(p, "PAUSED - [space] resume, [n] step");
                cv::imshow(kWindow, p);
            }
            break;
        case 'n': stepOnce = true; break;
        case 'h': showHud = !showHud; break;
        case 'p': showPanel = !showPanel; break;
        case 'r': showRadar = !showRadar; break;
        case 'v': showVoronoi = !showVoronoi; break;
        case 't': showTrails = !showTrails; break;
        case 'm': showHeatmap = !showHeatmap; break;
        case 'l': showLine = !showLine; break;
        case 's':
            if (!lastCanvas.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(cfg.outputDir, ec);
                const auto path = cfg.outputDir / ("snap_" + std::to_string(lastIndex) + ".png");
                cv::imwrite(path.string(), lastCanvas);
                log::info("Snapshot saved -> {}", path.string());
            }
            break;
        default: break;
    }
}

bool TrackingApp::Impl::windowClosed() const {
    return cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) < 1;
}

void TrackingApp::Impl::openOutputs() {
    if (!cfg.save.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.save.parent_path(), ec);
        writer.open(cfg.save.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), reader->fps(), canvasSize);
        if (!writer.isOpened()) throw std::runtime_error("Cannot open video writer: " + cfg.save.string());
        log::info("Recording -> {}", cfg.save.string());
    }
    if (!cfg.tracksCsv.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.tracksCsv.parent_path(), ec);
        csv.open(cfg.tracksCsv);
        csv << "frame,time_s,track_id,label,team,conf,x1,y1,x2,y2,ground_x_m,ground_y_m,speed_kmh,distance_m\n";
        log::info("Exporting tracks -> {}", cfg.tracksCsv.string());
    }
}

int TrackingApp::Impl::run() {
    log::info("Warming up detector ...");
    const auto t0 = std::chrono::steady_clock::now();
    detector->warmup({reader->width(), reader->height()}, cfg.ball && cfg.ballRoi ? 640 : 0);
    log::info("Warm-up done in {:.2f}s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    openOutputs();
    if (cfg.show) {
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(kWindow, canvasSize.width, canvasSize.height);
    }
    sysmon.start();
    reader->start();

    // Detector thread: run inference on frame N+1 while the main loop handles frame N.
    BoundedQueue<DetPacket> detQueue(2);
    std::atomic<bool> detError{false};
    std::thread detThread([&] {
        while (true) {
            auto packet = reader->read();
            if (!packet) break;
            const auto td = std::chrono::steady_clock::now();
            DetPacket out;
            try {
                out.detections = detector->detect(packet->frame, detectClasses, cfg.imgSize);
            } catch (const std::exception& e) {
                log::error("detector thread: {}", e.what());
                detError = true;
                break;
            }
            out.detectMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
            out.frame = std::move(*packet);
            if (!detQueue.push(std::move(out))) break;
        }
        detQueue.close();
    });

    perf = PerfMonitor{};
    while (running) {
        if (paused && !stepOnce) {
            const auto tp = std::chrono::steady_clock::now();
            handleKey(cv::waitKey(50) & 0xFF);
            perf.addPause(std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
            if (windowClosed()) break;
            continue;
        }
        stepOnce = false;

        std::optional<DetPacket> item;
        {
            auto t = perf.measure("wait");
            item = detQueue.pop();
        }
        if (!item) { log::info("End of stream"); break; }

        process(*item);
        perf.tick();

        if (cfg.show) {
            handleKey(cv::waitKey(1) & 0xFF);
            if (windowClosed()) { log::info("Window closed"); break; }
        } else if (perf.frames() % 30 == 0) {
            log::info("frame {} | {:.1f} FPS | det {:.1f} ms | players {} | ball {} | tilt {:.0f} H {:.1f}",
                      item->frame.index + 1, perf.fps(), perf.mean("detect"), activeTracks,
                      ball->visible() ? "yes" : "no", ground->tiltDeg(), ground->camHeight());
        }
        if (cfg.maxFrames && perf.frames() >= cfg.maxFrames) { log::info("Reached --max-frames={}", cfg.maxFrames); break; }
    }

    reader->stop();
    detQueue.close();
    if (detThread.joinable()) detThread.join();
    sysmon.stop();
    if (writer.isOpened()) writer.release();
    if (csv.is_open()) csv.close();
    if (cfg.show) cv::destroyAllWindows();
    report();
    return 0;
}

json TrackingApp::Impl::footballStats(double minSeconds) const {
    const double fps = reader->fps();
    json players = json::array();
    std::vector<std::tuple<std::string, std::optional<int>, double, double, double, double>> rows;
    for (const auto& [key, st] : kin->all()) {
        const double seconds = (st.lastFrame - st.firstFrame + 1) / fps;
        if (seconds < minSeconds) continue;
        const auto team = teams ? teams->teamOf(std::stoi(key)) : std::nullopt;
        const double avg = seconds > 0 ? st.distance / seconds * 3.6 : 0.0;
        rows.emplace_back(key, team, seconds, st.distance, st.maxSpeedKmh, avg);
    }
    std::ranges::sort(rows, [](auto& a, auto& b) { return std::get<3>(a) > std::get<3>(b); });
    for (auto& [key, team, seconds, dist, maxv, avg] : rows) {
        players.push_back({{"track_id", key}, {"team", teams ? teams->label(team) : ""},
                           {"seconds_tracked", std::round(seconds * 10) / 10}, {"distance_m", std::round(dist * 10) / 10},
                           {"max_speed_kmh", std::round(maxv * 10) / 10}, {"avg_speed_kmh", std::round(avg * 10) / 10}});
    }

    json teamsJson = json::object();
    if (teams && teams->ready()) {
        long possTotal = 0;
        for (auto [t, n] : possession) possTotal += n;
        for (int t = 0; t < 2; ++t) {
            const std::string letter(1, static_cast<char>('A' + t));
            double totalDist = 0, topSpeed = 0;
            std::string topTrack;
            int count = 0;
            for (auto& p : players) {
                if (p["team"] == letter) {
                    ++count;
                    totalDist += p["distance_m"].get<double>();
                    if (p["max_speed_kmh"].get<double>() > topSpeed) { topSpeed = p["max_speed_kmh"]; topTrack = p["track_id"]; }
                }
            }
            const cv::Vec3b c = teams->color(t);
            teamsJson[letter] = {{"name", teams->name(t)}, {"color_bgr", {c[0], c[1], c[2]}}, {"track_ids", count},
                                 {"total_distance_m", std::round(totalDist * 10) / 10}, {"top_speed_kmh", topSpeed},
                                 {"top_speed_track", topTrack},
                                 {"avg_space_control", controlSamples ? std::round(controlSum.at(t) / controlSamples * 1000) / 1000 : 0.0},
                                 {"possession", possTotal ? std::round(static_cast<double>(possession.count(t) ? possession.at(t) : 0) / possTotal * 1000) / 1000 : 0.0}};
        }
    }

    const MotionState* b = ballKin->find("ball");
    json ballJson = {{"detected_frames", ballFrames},
                     {"detected_ratio", std::round(static_cast<double>(ballFrames) / std::max(1, perf.frames()) * 1000) / 1000},
                     {"roi_recoveries", ballRoiHits},
                     {"distance_m", b ? std::round(b->distance * 10) / 10 : 0.0},
                     {"max_speed_kmh", b ? std::round(b->maxSpeedKmh * 10) / 10 : 0.0}};

    const int nFrames = std::max(1, perf.frames());
    json sourceJson = {{"fps", reader->fps()}, {"fps_source", reader->fpsSource()},
                       {"width", reader->width()}, {"height", reader->height()},
                       {"shot_cuts", shotCuts}, {"non_pitch_frames", nonPitchFrames},
                       {"lower_turf_ratio_mean", std::round(turfSum / nFrames * 1000) / 1000},
                       {"warnings", sourceWarnings()}};

    return {{"players", players}, {"teams", teamsJson}, {"ball", ballJson}, {"source", sourceJson},
            {"camera_model", {{"hfov_deg", cfg.hfovDeg}, {"person_height_m", cfg.personHeight},
                              {"tilt_deg", std::round(ground->tiltDeg() * 10) / 10}, {"height_m", std::round(ground->camHeight() * 100) / 100},
                              {"fit_residual", ground->ready() ? std::round(ground->residual() * 1000) / 1000 : 0.0}}},
            {"note", "Metric values come from a monocular self-calibrated camera model: expect roughly +/-20%."}};
}

void TrackingApp::Impl::report() {
    const std::string bar(78, '=');
    std::printf("\n%s\n Performance report\n%s\n", bar.c_str(), bar.c_str());
    std::printf("Frames: %d | avg FPS: %.2f | bottleneck: %s\n", perf.frames(), perf.avgFps(), perf.bottleneck().c_str());
    std::printf("%-10s%9s%9s%9s%9s%9s  (ms)\n", "stage", "mean", "p50", "p95", "p99", "max");
    for (const auto& st : {"decode", "detect", "camera", "track", "analytics", "ball", "render", "wait"}) {
        auto it = perf.stages().find(st);
        if (it == perf.stages().end()) continue;
        const auto s = it->second.summary();
        std::printf("%-10s%9.1f%9.1f%9.1f%9.1f%9.1f\n", st, s.at("mean"), s.at("p50"), s.at("p95"), s.at("p99"), s.at("max"));
    }
    const auto pk = sysmon.peaks();
    std::printf("Peaks: cpu=%.0f%% rss=%.0fMB gpu=%.0f%% vram=%.0fMB\n", pk.cpuPercent, pk.rssMb, pk.gpuUtil, pk.gpuMemMb);

    const json stats = footballStats();
    std::printf("%s\n Football stats\n%s\n", bar.c_str(), bar.c_str());
    for (auto& [letter, t] : stats["teams"].items()) {
        std::printf("Team %s: %d track IDs | distance %.0f m | top %.1f km/h (#%s) | space %.0f%% | possession %.0f%%\n",
                    t["name"].get<std::string>().c_str(), t["track_ids"].get<int>(), t["total_distance_m"].get<double>(),
                    t["top_speed_kmh"].get<double>(), t["top_speed_track"].get<std::string>().c_str(),
                    t["avg_space_control"].get<double>() * 100, t["possession"].get<double>() * 100);
    }
    const auto& b = stats["ball"];
    std::printf("Ball: detected %.0f%% of frames (%d ROI) | distance %.0f m | max %.0f km/h\n",
                b["detected_ratio"].get<double>() * 100, b["roi_recoveries"].get<int>(),
                b["distance_m"].get<double>(), b["max_speed_kmh"].get<double>());
    std::printf("%5s %4s %7s %7s %9s %9s\n", "id", "team", "time s", "dist m", "max km/h", "avg km/h");
    int shown = 0;
    for (auto& p : stats["players"]) {
        if (shown++ >= 15) break;
        std::printf("%5s %4s %7.1f %7.1f %9.1f %9.1f\n", p["track_id"].get<std::string>().c_str(),
                    p["team"].get<std::string>().c_str(), p["seconds_tracked"].get<double>(),
                    p["distance_m"].get<double>(), p["max_speed_kmh"].get<double>(), p["avg_speed_kmh"].get<double>());
    }
    const auto& src = stats["source"];
    std::printf("Source: %.2f fps (%s) | shot cuts %d | non-pitch frames %d | mean pitch turf %.0f%%\n",
                src["fps"].get<double>(), src["fps_source"].get<std::string>().c_str(), src["shot_cuts"].get<int>(),
                src["non_pitch_frames"].get<int>(), src["lower_turf_ratio_mean"].get<double>() * 100);
    for (const auto& w : src["warnings"]) std::printf("WARNING: %s\n", w.get<std::string>().c_str());
    std::printf("%s\n", bar.c_str());

    auto writeJson = [](const std::filesystem::path& path, const json& j) {
        if (path.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        std::ofstream(path) << j.dump(2);
    };
    if (!cfg.statsJson.empty()) { writeJson(cfg.statsJson, stats); log::info("Stats -> {}", cfg.statsJson.string()); }
    if (!cfg.perfJson.empty()) {
        json perfJson;
        perfJson["frames"] = perf.frames();
        perfJson["avg_fps"] = perf.avgFps();
        perfJson["bottleneck"] = perf.bottleneck();
        json stagesJson = json::object();
        for (const auto& [name, stat] : perf.stages()) stagesJson[name] = stat.summary();
        perfJson["stages_ms"] = stagesJson;
        perfJson["device"] = detector->provider();
        perfJson["gpu"] = sysmon.gpuName();
        perfJson["source"] = {{"width", reader->width()}, {"height", reader->height()}, {"fps", reader->fps()},
                              {"frames", reader->frameCount()}};
        writeJson(cfg.perfJson, perfJson);
        log::info("Perf -> {}", cfg.perfJson.string());
    }
}

// ---------------------------------------------------------------------------- public wrapper

TrackingApp::TrackingApp(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
TrackingApp::~TrackingApp() = default;
int TrackingApp::run() { return impl_->run(); }

}  // namespace ft
