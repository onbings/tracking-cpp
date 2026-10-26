"""Build output/cut_test.mp4: an edited clip with two shot cuts and a non-pitch close-up.

Used to validate shot-cut and non-pitch-view detection (DOCUMENTATION.md section 15.1):
  frames   0-89  : wide drone view (reference frames 400-489)
  frames  90-134 : close-up of the wall/banner (top band of reference frames 1500-1544)
  frames 135-224 : another part of the match, lower and closer (last 200 frames of the reference)
Expected: cuts at frames 90 and 135, 45 non-pitch frames.

Usage:  python tools/make_cut_test.py [source.mp4]
"""
import pathlib
import sys

import cv2

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:\media\14369070-uhd_3840_2160_30fps.mp4"
OUT = pathlib.Path(__file__).resolve().parents[1] / "output" / "cut_test.mp4"
W, H = 1920, 1080

cap = cv2.VideoCapture(SRC)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
OUT.parent.mkdir(exist_ok=True)
out = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))


def segment(start, count, crop=lambda f: f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(crop(frame), (W, H), interpolation=cv2.INTER_AREA))


segment(400, 90)
segment(1500, 45, lambda f: f[: f.shape[0] // 4, f.shape[1] // 3: f.shape[1] // 3 + f.shape[1] // 4])
segment(n - 200, 90)
out.release()
print(f"wrote {OUT}")
