"""
hand_tracker.py  –  MediaPipe Hands wrapper with robust error handling.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import get_config
from logger import get_logger

log = get_logger(__name__)

# ── Landmark indices ───────────────────────────────────────────────────────────
WRIST       = 0
THUMB_TIP   = 4
INDEX_MCP   = 5;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_TIP = 12
RING_MCP    = 13; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_TIP  = 20


@dataclass
class HandData:
    """Processed data for one detected hand in a frame."""
    landmarks:    np.ndarray       # (21, 3) normalised
    landmarks_px: np.ndarray       # (21, 2) pixel coords
    handedness:   str              # "Left" | "Right"
    fingers_up:   list[bool]       # [thumb, index, middle, ring, pinky]
    index_tip_px: tuple[int, int]  # convenience: pixel tip of index finger
    confidence:   float            # MediaPipe detection confidence


class HandTracker:
    """
    Wraps MediaPipe Hands with error handling and optimised landmark access.

    Usage
    -----
        tracker = HandTracker()
        hands = tracker.process(bgr_frame)
        if hands:
            tip = hands[0].index_tip_px
    """

    def __init__(
        self,
        max_hands:             int   = 1,
        detection_confidence:  float = 0.75,
        tracking_confidence:   float = 0.75,
    ) -> None:
        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_style  = mp.solutions.drawing_styles
        self._closed    = False

        try:
            self.hands = self._mp_hands.Hands(
                static_image_mode        = False,
                max_num_hands            = max_hands,
                min_detection_confidence = detection_confidence,
                min_tracking_confidence  = tracking_confidence,
            )
            log.info("MediaPipe Hands initialised (max=%d, det=%.2f, trk=%.2f)",
                     max_hands, detection_confidence, tracking_confidence)
        except Exception as exc:
            log.critical("Failed to initialise MediaPipe Hands: %s", exc)
            raise

    # ── Public ─────────────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> list[HandData]:
        """
        Detect hands in *bgr_frame* and return a list of HandData objects.
        Returns [] on any MediaPipe error.
        """
        if self._closed:
            return []

        try:
            h, w = bgr_frame.shape[:2]
            rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
        except Exception as exc:
            log.warning("MediaPipe detection error: %s", exc)
            return []

        if not results.multi_hand_landmarks:
            return []

        hands: list[HandData] = []
        for lm_list, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            try:
                landmarks    = np.array([[l.x, l.y, l.z] for l in lm_list.landmark], np.float32)
                landmarks_px = (landmarks[:, :2] * [w, h]).astype(int)
                handedness   = handedness_info.classification[0].label
                confidence   = handedness_info.classification[0].score
                fingers_up   = self._fingers_up(landmarks, handedness)
                tip          = tuple(landmarks_px[INDEX_TIP].tolist())

                hands.append(HandData(
                    landmarks    = landmarks,
                    landmarks_px = landmarks_px,
                    handedness   = handedness,
                    fingers_up   = fingers_up,
                    index_tip_px = tip,
                    confidence   = confidence,
                ))
            except Exception as exc:
                log.warning("Landmark parse error: %s", exc)

        return hands

    def draw_landmarks(self, frame: np.ndarray, hand: HandData) -> np.ndarray:
        """Overlay hand skeleton on *frame* in-place and return it."""
        try:
            lm_list = self._build_landmark_list(hand.landmarks)
            self._mp_draw.draw_landmarks(
                frame, lm_list,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_style.get_default_hand_landmarks_style(),
                self._mp_style.get_default_hand_connections_style(),
            )
        except Exception as exc:
            log.debug("draw_landmarks error: %s", exc)
        return frame

    def release(self) -> None:
        if not self._closed:
            try:
                self.hands.close()
            except Exception:
                pass
            self._closed = True
            log.info("HandTracker released.")

    # ── Private ────────────────────────────────────────────────────────────────

    def _fingers_up(self, lm: np.ndarray, handedness: str) -> list[bool]:
        fingers = []
        if handedness == "Right":
            fingers.append(bool(lm[THUMB_TIP][0] < lm[THUMB_TIP - 1][0]))
        else:
            fingers.append(bool(lm[THUMB_TIP][0] > lm[THUMB_TIP - 1][0]))
        for tip, _ in [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                       (RING_TIP, RING_MCP),   (PINKY_TIP, PINKY_MCP)]:
            fingers.append(bool(lm[tip][1] < lm[tip - 2][1]))
        return fingers

    def _build_landmark_list(self, landmarks: np.ndarray):
        from mediapipe.framework.formats import landmark_pb2
        ll = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in landmarks:
            l = ll.landmark.add()
            l.x, l.y, l.z = float(x), float(y), float(z)
        return ll
