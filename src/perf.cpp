/**
 * @file perf.cpp
 * @brief Implementation of the performance and system monitors.
 *
 * GPU telemetry is read through nvml.dll loaded at runtime (no link-time CUDA dependency). If the
 * DLL or a symbol is missing, GPU fields stay disabled and the rest keeps working.
 */
#include "perf.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace ft {

void RollingStat::add(double value) {
    window_.push_back(value);
    if (window_.size() > kWindow) window_.pop_front();
    series_.push_back(value);
}

double RollingStat::mean() const {
    if (window_.empty()) return 0.0;
    return std::accumulate(window_.begin(), window_.end(), 0.0) / window_.size();
}

double RollingStat::last() const { return window_.empty() ? 0.0 : window_.back(); }

std::map<std::string, double> RollingStat::summary() const {
    if (series_.empty()) return {{"count", 0}, {"mean", 0}, {"p50", 0}, {"p95", 0}, {"p99", 0}, {"max", 0}};
    std::vector<double> sorted = series_;
    std::ranges::sort(sorted);
    auto pct = [&](double p) { return sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(p * sorted.size()))]; };
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    return {{"count", static_cast<double>(sorted.size())}, {"mean", mean}, {"p50", pct(0.50)},
            {"p95", pct(0.95)}, {"p99", pct(0.99)}, {"max", sorted.back()}};
}

void PerfMonitor::record(const std::string& stage, double ms) { stages_[stage].add(ms); }

void PerfMonitor::tick() {
    ticks_.push_back(std::chrono::steady_clock::now());
    if (ticks_.size() > 60) ticks_.pop_front();
    ++frames_;
    fpsHistory_.push_back(fps());
    if (fpsHistory_.size() > 240) fpsHistory_.pop_front();
}

void PerfMonitor::addPause(double seconds) {
    pausedSeconds_ += seconds;
    ticks_.clear();
}

double PerfMonitor::fps() const {
    if (ticks_.size() < 2) return 0.0;
    const double span = std::chrono::duration<double>(ticks_.back() - ticks_.front()).count();
    return span > 0 ? (ticks_.size() - 1) / span : 0.0;
}

double PerfMonitor::avgFps() const {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count() - pausedSeconds_;
    return elapsed > 0 ? frames_ / elapsed : 0.0;
}

double PerfMonitor::mean(const std::string& stage) const {
    auto it = stages_.find(stage);
    return it == stages_.end() ? 0.0 : it->second.mean();
}

std::string PerfMonitor::bottleneck() const {
    double mainSum = 0.0, worst = -1.0;
    std::string worstStage = "-";
    for (const auto& s : kPipeline) {
        if (s == "wait") continue;
        const double m = mean(s);
        mainSum += m;
        if (m > worst) { worst = m; worstStage = s; }
    }
    std::string best = "main/" + worstStage;
    double bestVal = mainSum;
    for (const auto& s : kConcurrent) {
        if (mean(s) > bestVal) { bestVal = mean(s); best = s; }
    }
    return best;
}

// ------------------------------------------------------------------ SystemMonitor

namespace {
// Minimal NVML binding loaded at runtime.
struct Nvml {
    using InitFn = int (*)();
    using ShutdownFn = int (*)();
    using HandleFn = int (*)(unsigned, void**);
    using NameFn = int (*)(void*, char*, unsigned);
    using UtilFn = int (*)(void*, unsigned* /*gpu*/ , unsigned* /*mem*/);
    using MemFn = int (*)(void*, unsigned long long*, unsigned long long*, unsigned long long*);
    using TempFn = int (*)(void*, int, unsigned*);
    using PowerFn = int (*)(void*, unsigned*);

#ifdef _WIN32
    HMODULE lib = nullptr;
#endif
    void* handle = nullptr;
    UtilFn utilization = nullptr;
    MemFn memory = nullptr;
    TempFn temperature = nullptr;
    PowerFn power = nullptr;
    ShutdownFn shutdown = nullptr;
    bool ok = false;
    std::string name;
};
}  // namespace

struct SystemMonitor::Impl {
    std::thread thread;
    std::atomic<bool> stop{false};
    mutable std::mutex mutex;
    SystemSnapshot current;
    SystemSnapshot peak;
    Nvml nvml;

    void sample() {
        SystemSnapshot s;
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            s.rssMb = pmc.WorkingSetSize / (1024.0 * 1024.0);
        }
        static ULARGE_INTEGER lastCpu{}, lastSys{};
        FILETIME ftCreate, ftExit, ftKernel, ftUser, ftNow;
        if (GetProcessTimes(GetCurrentProcess(), &ftCreate, &ftExit, &ftKernel, &ftUser)) {
            GetSystemTimeAsFileTime(&ftNow);
            ULARGE_INTEGER cpu, sys, nowK{}, nowU{}, nowS{};
            nowK.LowPart = ftKernel.dwLowDateTime; nowK.HighPart = ftKernel.dwHighDateTime;
            nowU.LowPart = ftUser.dwLowDateTime; nowU.HighPart = ftUser.dwHighDateTime;
            nowS.LowPart = ftNow.dwLowDateTime; nowS.HighPart = ftNow.dwHighDateTime;
            cpu.QuadPart = nowK.QuadPart + nowU.QuadPart;
            sys.QuadPart = nowS.QuadPart;
            if (lastSys.QuadPart != 0 && sys.QuadPart > lastSys.QuadPart) {
                const double busy = static_cast<double>(cpu.QuadPart - lastCpu.QuadPart);
                const double wall = static_cast<double>(sys.QuadPart - lastSys.QuadPart);
                const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
                s.cpuPercent = 100.0 * busy / wall / cores;
            }
            lastCpu = cpu;
            lastSys = sys;
        }
#endif
        if (nvml.ok) {
            unsigned gpu = 0, mem = 0, temp = 0, power = 0;
            unsigned long long total = 0, used = 0, free = 0;
            if (nvml.utilization && nvml.utilization(nvml.handle, &gpu, &mem) == 0) s.gpuUtil = gpu;
            if (nvml.memory && nvml.memory(nvml.handle, &total, &free, &used) == 0) {
                s.gpuMemMb = used / (1024.0 * 1024.0);
                s.gpuMemTotalMb = total / (1024.0 * 1024.0);
            }
            if (nvml.temperature && nvml.temperature(nvml.handle, 0, &temp) == 0) s.gpuTempC = temp;
            if (nvml.power && nvml.power(nvml.handle, &power) == 0) s.gpuPowerW = power / 1000.0;
            s.hasGpu = true;
        }
        std::scoped_lock lock(mutex);
        current = s;
        peak.cpuPercent = std::max(peak.cpuPercent, s.cpuPercent);
        peak.rssMb = std::max(peak.rssMb, s.rssMb);
        peak.gpuUtil = std::max(peak.gpuUtil, s.gpuUtil);
        peak.gpuMemMb = std::max(peak.gpuMemMb, s.gpuMemMb);
        peak.hasGpu = peak.hasGpu || s.hasGpu;
    }
};

SystemMonitor::SystemMonitor() : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    Nvml& n = impl_->nvml;
    n.lib = LoadLibraryA("nvml.dll");
    if (!n.lib) n.lib = LoadLibraryA("C:/Windows/System32/nvml.dll");
    if (n.lib) {
        auto init = reinterpret_cast<Nvml::InitFn>(GetProcAddress(n.lib, "nvmlInit_v2"));
        auto getHandle = reinterpret_cast<Nvml::HandleFn>(GetProcAddress(n.lib, "nvmlDeviceGetHandleByIndex_v2"));
        auto getName = reinterpret_cast<Nvml::NameFn>(GetProcAddress(n.lib, "nvmlDeviceGetName"));
        n.utilization = reinterpret_cast<Nvml::UtilFn>(GetProcAddress(n.lib, "nvmlDeviceGetUtilizationRates"));
        n.memory = reinterpret_cast<Nvml::MemFn>(GetProcAddress(n.lib, "nvmlDeviceGetMemoryInfo"));
        n.temperature = reinterpret_cast<Nvml::TempFn>(GetProcAddress(n.lib, "nvmlDeviceGetTemperature"));
        n.power = reinterpret_cast<Nvml::PowerFn>(GetProcAddress(n.lib, "nvmlDeviceGetPowerUsage"));
        n.shutdown = reinterpret_cast<Nvml::ShutdownFn>(GetProcAddress(n.lib, "nvmlShutdown"));
        if (init && getHandle && init() == 0 && getHandle(0, &n.handle) == 0) {
            n.ok = true;
            char name[96] = {0};
            if (getName && getName(n.handle, name, sizeof(name)) == 0) gpuName_ = name;
        }
    }
#endif
}

SystemMonitor::~SystemMonitor() {
    stop();
#ifdef _WIN32
    if (impl_->nvml.shutdown) impl_->nvml.shutdown();
    if (impl_->nvml.lib) FreeLibrary(impl_->nvml.lib);
#endif
}

void SystemMonitor::start() {
    impl_->sample();  // prime the CPU delta
    impl_->thread = std::thread([this] {
        while (!impl_->stop.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (impl_->stop.load()) break;
            impl_->sample();
        }
    });
}

void SystemMonitor::stop() {
    impl_->stop.store(true);
    if (impl_->thread.joinable()) impl_->thread.join();
}

SystemSnapshot SystemMonitor::snapshot() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->current;
}

SystemSnapshot SystemMonitor::peaks() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->peak;
}

}  // namespace ft
