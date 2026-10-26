# Football Tracker — C++20 port

Real-time football analytics on moving-camera 4K video: **OpenCV** decoding → **YOLO11** detection
(ONNX Runtime, CUDA) → **BoT-SORT** tracking → team classification, speed/distance in metres, ball
tracking, and Voronoi space control. A C++20 re-implementation of the Python proof of concept in
`C:\pro\tracking`.

See **[DOCUMENTATION.md](DOCUMENTATION.md)** for the full design and the differences from the Python version.

## Prerequisites

- Windows x64, **Visual Studio 2022** (v143 toolset)
- **CMake ≥ 3.25**, **Ninja**, **vcpkg** at `C:\pro\vcpkg`
- An NVIDIA GPU with a recent driver (CUDA execution provider). CPU works too (`--device cpu`).

## Setup

```powershell
# 1. Prepare the neighboring Python project and its exporter environment (one time)
cd C:\pro\tracking
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 2. Download the source checkpoint into C:\pro\tracking\weights
#    Use a video that exists on your machine. Repeat with yolo11n/yolo11m if you want those models too.
.\.venv\Scripts\python.exe -m tracker --model yolo11s --device cpu --source C:\path\to\video.mp4 --max-frames 1

# 3. Export static ONNX models (writes four files to tracking-cpp\models\)
cd C:\pro\tracking-cpp
C:\pro\tracking\.venv\Scripts\python.exe tools\export_static.py yolo11s

# 4. Fetch CUDA 12.9 + cuDNN 9 runtime DLLs (no admin rights or full CUDA Toolkit required)
powershell -ExecutionPolicy Bypass -File tools\fetch_cuda_runtime.ps1

# 5. Configure and build (vcpkg manifest mode; the first configure builds dependencies)
cmake --preset default
cmake --build build --config Release

# 6. Run
build\Release\tracker.exe
```

The C++ repository intentionally does not track generated `.onnx` files or CUDA runtime DLLs.
The Python tracker downloads the requested Ultralytics `.pt` checkpoint to
`C:\pro\tracking\weights`; `tools/export_static.py` reads it from that fixed path and creates
FP32/FP16 models for 1280 and 640 input sizes. FP16 export requires a CUDA-capable NVIDIA GPU.
The CUDA fetch script downloads the redistributable NVIDIA wheels into `third_party\wheels` and
extracts their DLLs into `third_party\cuda-runtime\bin`. If your Python repository or weights
folder is elsewhere, update the `WEIGHTS` path in `tools/export_static.py` before exporting.
To support `--model yolo11n` or `yolo11m`, first run the Python download step with that model, then
pass all desired names to the exporter, for example `tools\export_static.py yolo11n yolo11s yolo11m`.

In VS Code: install the recommended extensions, pick the **default** CMake preset, and use the
**build**/**test** tasks or the launch configurations (F5).

## Usage

```powershell
build\Release\tracker.exe                                   # window with HUD
build\Release\tracker.exe --voronoi --save output\tracked.mp4 --stats-json output\stats.json
build\Release\tracker.exe --no-show --max-frames 300        # headless benchmark
build\Release\tracker.exe --device cpu --model yolo11n      # CPU fallback
build\Release\tracker.exe --source other.mp4 --hfov 60 --fps 25   # another clip: set its lens FOV / frame rate
build\Release\tracker.exe --help
```

**Other clips.** Any football video works for detection and tracking. Speed, distance and space
control assume an elevated, wide, continuous shot (drone or tactical camera). Shot cuts, close-ups
and broken frame-rate metadata are detected automatically and reported as warnings. See
[DOCUMENTATION.md §29.1](DOCUMENTATION.md#29-known-limitations) for the limits.

Keys: `q` quit · `space` pause · `n` step · `h` HUD · `p` panel · `r` radar · `v` Voronoi ·
`t` trails · `m` heatmap · `l` line · `s` snapshot.

## What changed from the Python version

- **Inference:** ONNX Runtime (CUDA EP) instead of PyTorch/ultralytics; YOLO11 exported to ONNX.
- **Pruned** the back-ends the Python benchmarks showed to be useless or slow here: **YOLOv5**,
  **DeepSORT**, **ByteTrack**, and **ReID**. Only YOLO11 + BoT-SORT remain.
- **Self-contained:** the tracker, Kalman filters, team clustering and all analytics are hand-written
  C++20 (no ultralytics internals), with Doxygen comments on every class and method.
- See [DOCUMENTATION.md](DOCUMENTATION.md) for the complete list.
