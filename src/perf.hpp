/**
 * @file perf.hpp
 * @brief Per-stage latency and throughput measurement, plus optional system telemetry.
 */
#pragma once

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ft {

/**
 * @brief A short rolling window (for the live HUD) plus the full series (for the final report).
 */
class RollingStat {
public:
    /// @brief Appends a sample (milliseconds).
    void add(double value);
    [[nodiscard]] double mean() const;                 ///< Mean over the window.
    [[nodiscard]] double last() const;                 ///< Most recent sample.
    /// @return {count, mean, p50, p95, p99, max} over the full series.
    [[nodiscard]] std::map<std::string, double> summary() const;

private:
    std::deque<double> window_;
    std::vector<double> series_;
    static constexpr std::size_t kWindow = 60;
};

/**
 * @brief Tracks per-stage latency and FPS, and finds the bottleneck.
 *
 * Stages run on the main thread; decode and detect run concurrently on their own threads and only
 * limit FPS when they exceed the main-thread total.
 */
class PerfMonitor {
public:
    /// @brief RAII timer: records the elapsed time into @p stage on destruction.
    class Scope {
    public:
        Scope(PerfMonitor& monitor, std::string stage)
            : monitor_(monitor), stage_(std::move(stage)), start_(std::chrono::steady_clock::now()) {}
        ~Scope() {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start_).count();
            monitor_.record(stage_, ms);
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        PerfMonitor& monitor_;
        std::string stage_;
        std::chrono::steady_clock::time_point start_;
    };

    /// @return an RAII timer for @p stage.
    [[nodiscard]] Scope measure(const std::string& stage) { return Scope(*this, stage); }
    /// @brief Records a duration (ms) for a stage directly (e.g. for concurrent stages).
    void record(const std::string& stage, double ms);
    /// @brief Marks one processed frame (updates FPS).
    void tick();
    /// @brief Excludes paused wall-clock time from throughput.
    void addPause(double seconds);

    [[nodiscard]] double fps() const;                       ///< Instantaneous FPS (last 60 frames).
    [[nodiscard]] double avgFps() const;                    ///< Average FPS since start (minus pauses).
    [[nodiscard]] int frames() const { return frames_; }    ///< Frames processed.
    [[nodiscard]] double mean(const std::string& stage) const;  ///< Window mean of a stage (0 if absent).
    [[nodiscard]] std::string bottleneck() const;           ///< Name of the limiting stage.
    [[nodiscard]] const std::deque<double>& fpsHistory() const { return fpsHistory_; }  ///< For the sparkline.
    [[nodiscard]] const std::map<std::string, RollingStat>& stages() const { return stages_; }

    /// Main-thread stages, in pipeline order.
    static inline const std::vector<std::string> kPipeline{"wait", "camera", "track", "analytics", "ball", "render"};
    /// Stages measured on worker threads.
    static inline const std::vector<std::string> kConcurrent{"decode", "detect"};

private:
    std::map<std::string, RollingStat> stages_;
    std::deque<std::chrono::steady_clock::time_point> ticks_;
    std::deque<double> fpsHistory_;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    double pausedSeconds_ = 0.0;
    int frames_ = 0;
};

/**
 * @brief One periodic snapshot of CPU/RAM/GPU usage.
 */
struct SystemSnapshot {
    double cpuPercent = 0;   ///< Process CPU usage (% of one core-equivalent average).
    double rssMb = 0;        ///< Resident set size (MB).
    bool hasGpu = false;     ///< GPU telemetry available.
    double gpuUtil = 0;      ///< GPU utilisation (%).
    double gpuMemMb = 0;     ///< GPU memory used (MB).
    double gpuMemTotalMb = 0;///< GPU memory total (MB).
    double gpuTempC = 0;     ///< GPU temperature (°C).
    double gpuPowerW = 0;    ///< GPU power draw (W).
};

/**
 * @brief Samples system/GPU usage on a background thread (via nvml.dll if present).
 */
class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();
    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    void start();                                   ///< Begin sampling (1 Hz).
    void stop();                                    ///< Stop sampling.
    [[nodiscard]] SystemSnapshot snapshot() const;  ///< Latest snapshot (thread-safe copy).
    [[nodiscard]] SystemSnapshot peaks() const;     ///< Peak values seen.
    [[nodiscard]] const std::string& gpuName() const { return gpuName_; }  ///< GPU name (may be empty).

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string gpuName_;
};

}  // namespace ft
