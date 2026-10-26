// Micro-benchmark: split YOLO detect() into preprocess / inference / postprocess on the real video.
// Built only when configured; not part of the app.
#include "detector.hpp"
#include "runtime_env.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    ft::prepareCudaRuntime({});
    const std::string model = argc > 1 ? argv[1] : "yolo11s";
    ft::YoloDetector det("C:/pro/tracking-cpp/models", model, "auto", true, 1280, 0.1f, 0.5f);
    std::printf("provider %s fp16 %d\n", det.provider().c_str(), det.fp16());

    cv::VideoCapture cap("C:/media/14369070-uhd_3840_2160_30fps.mp4");
    cap.set(cv::CAP_PROP_POS_FRAMES, 400);
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 40; ++i) { cv::Mat f; cap.read(f); frames.push_back(f); }

    for (int i = 0; i < 5; ++i) det.detect(frames[0], {}, 1280);  // warmup

    const std::vector<int> cls{};
    auto t0 = clk::now();
    int total = 0;
    for (auto& f : frames) total += (int)det.detect(f, cls, 1280).size();
    auto t1 = clk::now();
    std::printf("full detect: %.1f ms/frame (%d dets)\n", ms(t0, t1) / frames.size(), total);

    // Pure session.Run() cost (no preprocess/postprocess).
    t0 = clk::now();
    double runSum = 0;
    for (auto& f : frames) { det.detect(f, cls, 1280); runSum += det.lastRunMs(); }
    t1 = clk::now();
    std::printf("session.Run only: %.1f ms/frame\n", runSum / frames.size());

    // Preprocess-only cost (resize+cvt+normalize) by timing a bare resize pipeline.
    t0 = clk::now();
    for (auto& f : frames) {
        cv::Mat r, rgb;
        cv::resize(f, r, {1280, 720}, 0, 0, cv::INTER_AREA);
        cv::cvtColor(r, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
        cv::Mat half;
        rgb.convertTo(half, CV_16F);
    }
    t1 = clk::now();
    std::printf("preprocess only: %.1f ms/frame\n", ms(t0, t1) / frames.size());
    return 0;
}
