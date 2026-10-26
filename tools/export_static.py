"""Export static-shape YOLO11 ONNX models for fast ONNX Runtime CUDA inference.

Dynamic input axes let one model serve several sizes but prevent ONNX Runtime from fully planning
the CUDA graph, which is much slower. We instead export one static model per input size actually
used: 1280 (full frame, letterboxed to 1280x1280) and 640 (ball ROI).

    C:\\pro\\tracking\\.venv\\Scripts\\python.exe tools\\export_static.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\pro\tracking\weights")
OUT = ROOT / "models"
SIZES = {1280: "", 640: "_640"}


def export(name: str) -> None:
    from ultralytics import YOLO

    OUT.mkdir(exist_ok=True)
    for size, suffix in SIZES.items():
        for half in (False, True):
            model = YOLO(str(WEIGHTS / f"{name}.pt"))
            path = Path(model.export(format="onnx", dynamic=False, half=half, simplify=True, imgsz=size,
                                     device=0 if half else "cpu", opset=17))
            tag = ("_fp16" if half else "") 
            target = OUT / f"{name}{suffix}{tag}.onnx"
            shutil.move(str(path), target)
            print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    for n in sys.argv[1:] or ["yolo11n", "yolo11s", "yolo11m"]:
        export(n)
