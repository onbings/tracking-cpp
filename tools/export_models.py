"""Export the YOLO11 detectors used by tracking-cpp to ONNX.

Run with the Python PoC environment (it has ultralytics + torch):

    C:\\pro\\tracking\\.venv\\Scripts\\python.exe tools\\export_models.py            # yolo11n/s/m
    C:\\pro\\tracking\\.venv\\Scripts\\python.exe tools\\export_models.py yolo11s    # one model

For each model two files are written to ``models/``:

* ``<name>.onnx``       FP32, dynamic batch/height/width -> used on CPU
* ``<name>_fp16.onnx``  FP16, dynamic height/width       -> used with the CUDA execution provider

Dynamic spatial axes let the same model run on the full frame (e.g. 1280x736) and on the
640x640 ball ROI without re-exporting.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")  # old-driver workaround, see DOCUMENTATION.md

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\pro\tracking\weights")
OUT = ROOT / "models"


def export(name: str) -> None:
    from ultralytics import YOLO

    OUT.mkdir(exist_ok=True)
    pt = WEIGHTS / f"{name}.pt"
    for half in (False, True):
        model = YOLO(str(pt))
        path = Path(model.export(format="onnx", dynamic=True, half=half, simplify=True, imgsz=1280,
                                 device=0 if half else "cpu", opset=17))
        target = OUT / (f"{name}_fp16.onnx" if half else f"{name}.onnx")
        shutil.move(str(path), target)
        print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["yolo11n", "yolo11s", "yolo11m"]:
        export(n)
