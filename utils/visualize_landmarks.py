"""
utils/visualize_landmarks.py  –  Debug landmark indices & skeleton overlay.

Usage
-----
    python utils/visualize_landmarks.py [--camera 0]
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from config import get_config
from logger import setup_logging
from hand_tracker import HandTracker

setup_logging()

# Finger colours for skeleton segments
FINGER_COLORS = {
    "thumb":  (0, 128, 255),
    "index":  (0, 255, 0),
    "middle": (255, 255, 0),
    "ring":   (255, 0, 255),
    "pinky":  (0, 200, 255),
}
CONNECTIONS = [
    # Thumb
    (0,1,"thumb"),(1,2,"thumb"),(2,3,"thumb"),(3,4,"thumb"),
    # Index
    (5,6,"index"),(6,7,"index"),(7,8,"index"),
    # Middle
    (9,10,"middle"),(10,11,"middle"),(11,12,"middle"),
    # Ring
    (13,14,"ring"),(14,15,"ring"),(15,16,"ring"),
    # Pinky
    (17,18,"pinky"),(18,19,"pinky"),(19,20,"pinky"),
    # Palm
    (0,5,"index"),(5,9,"middle"),(9,13,"ring"),(13,17,"pinky"),(0,17,"pinky"),
]


def draw_debug(frame: np.ndarray, hand) -> np.ndarray:
    h, w = frame.shape[:2]
    lm = hand.landmarks_px

    for a, b, finger in CONNECTIONS:
        cv2.line(frame, tuple(lm[a]), tuple(lm[b]),
                 FINGER_COLORS[finger], 2, cv2.LINE_AA)

    for i, pt in enumerate(lm):
        cv2.circle(frame, tuple(pt), 5, (255, 255, 255), -1)
        cv2.circle(frame, tuple(pt), 5, (0, 0, 0), 1)
        cv2.putText(frame, str(i), (pt[0]+6, pt[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    fingers = hand.fingers_up
    labels  = ["T", "I", "M", "R", "P"]
    for i, (l, up) in enumerate(zip(labels, fingers)):
        col = (0, 255, 80) if up else (50, 50, 50)
        cv2.circle(frame, (20 + i*24, h-30), 8, col, -1)
        cv2.putText(frame, l, (15+i*24, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    cv2.putText(frame, hand.handedness, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return frame


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=get_config().camera.index)
    args = p.parse_args()

    tracker = HandTracker()
    cap     = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Landmark Visualiser – press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        if hands:
            frame = draw_debug(frame, hands[0])
        cv2.imshow("Landmark Visualiser", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

    cap.release(); cv2.destroyAllWindows(); tracker.release()

if __name__ == "__main__":
    main()
