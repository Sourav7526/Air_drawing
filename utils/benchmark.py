"""
utils/benchmark.py  –  FPS, gesture latency, and camera latency profiler.

Usage
-----
    python utils/benchmark.py [--frames 300] [--camera 0]
"""

from __future__ import annotations
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from config import get_config
from logger import setup_logging, get_logger
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, Gesture

setup_logging()
log = get_logger(__name__)


def benchmark(num_frames: int = 300, camera_idx: int = 0) -> None:
    cfg       = get_config()
    tracker   = HandTracker()
    recognizer = GestureRecognizer()

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.critical("Cannot open camera %d", camera_idx); sys.exit(1)

    cam_times:     list[float] = []
    tracker_times: list[float] = []
    gesture_times: list[float] = []
    frame_times:   list[float] = []

    print(f"\nBenchmarking {num_frames} frames …")
    loop_start = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Camera read
        t_cam = time.perf_counter()
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        cam_times.append(time.perf_counter() - t_cam)

        # Hand tracking
        t_trk = time.perf_counter()
        hands = tracker.process(frame)
        tracker_times.append(time.perf_counter() - t_trk)

        # Gesture inference
        if hands:
            t_gest = time.perf_counter()
            recognizer.recognize(hands[0].landmarks, hands[0].fingers_up)
            gesture_times.append(time.perf_counter() - t_gest)

        frame_times.append(time.perf_counter() - t0)

        if (i + 1) % 50 == 0:
            recent = frame_times[-50:]
            fps    = 1.0 / (sum(recent) / len(recent))
            print(f"  Frame {i+1:4d}/{num_frames}  FPS≈{fps:.1f}")

    cap.release(); tracker.release()

    total = time.perf_counter() - loop_start

    def stats(lst: list[float], name: str) -> None:
        if not lst: return
        a = np.array(lst) * 1000   # ms
        print(f"  {name:<25s}  mean={a.mean():.2f}ms  p95={np.percentile(a,95):.2f}ms  max={a.max():.2f}ms")

    print(f"\n{'='*60}")
    print(f"  Frames processed : {len(frame_times)}")
    print(f"  Total time       : {total:.1f}s")
    print(f"  Overall FPS      : {len(frame_times)/total:.1f}")
    print()
    stats(cam_times,     "Camera read")
    stats(tracker_times, "MediaPipe tracking")
    stats(gesture_times, "Gesture inference")
    stats(frame_times,   "Full frame pipeline")
    print(f"{'='*60}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--camera", type=int, default=get_config().camera.index)
    args = p.parse_args()
    benchmark(args.frames, args.camera)

if __name__ == "__main__":
    main()
