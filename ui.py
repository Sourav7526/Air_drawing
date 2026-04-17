"""
ui.py  –  HUD overlay: palette, gesture panel, confidence, FPS graph, help.
"""

from __future__ import annotations

import cv2
import numpy as np
import time
from collections import deque

from canvas import PALETTE, PALETTE_NAMES
from gesture_recognizer import Gesture
from config import get_config
from logger import get_logger

log = get_logger(__name__)

# ── Colour constants ───────────────────────────────────────────────────────────
HUD_BG     = (20,  20,  20)
HUD_ACCENT = (0, 200, 255)
HUD_TEXT   = (230, 230, 230)
HUD_WARN   = (0, 100, 255)
HUD_OK     = (0, 220, 80)

GESTURE_LABELS = {
    Gesture.IDLE:            "Idle",
    Gesture.DRAW:            "Draw",
    Gesture.ERASE:           "Erase",
    Gesture.CLEAR:           "Clear",
    Gesture.COLOR_PICK:      "Color Pick",
    Gesture.UNDO:            "Undo",
    Gesture.REDO:            "Redo",
    Gesture.SAVE_CANVAS:     "Save",
    Gesture.BRUSH_SIZE_UP:   "Brush +",
    Gesture.BRUSH_SIZE_DOWN: "Brush -",
    Gesture.DRAW_RECTANGLE:  "Rectangle",
    Gesture.DRAW_CIRCLE:     "Circle",
}


class UI:
    """Stateful HUD renderer."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.w, self.h = frame_w, frame_h
        cfg             = get_config().ui
        self._cfg       = cfg
        self._fps_buf:   deque[float] = deque(maxlen=cfg.fps_graph_length)
        self._status_msg            = ""
        self._status_until: float   = 0.0

    # ── Public ─────────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        *,
        current_color_idx: int,
        current_gesture:   Gesture,
        confidence:        float,
        brush_thickness:   int,
        eraser_size:       int,
        draw_mode:         str,
        fingers_up:        list[bool] | None = None,
    ) -> np.ndarray:
        self._fps_buf.append(time.time())

        frame = self._draw_palette(frame, current_color_idx)
        frame = self._draw_info_panel(frame, current_gesture, confidence,
                                      brush_thickness, eraser_size, draw_mode)
        frame = self._draw_fps(frame)

        if self._cfg.show_fps_graph:
            frame = self._draw_fps_graph(frame)

        if self._status_msg and time.time() < self._status_until:
            frame = self._draw_status(frame)

        if self._cfg.show_finger_debug and fingers_up is not None:
            frame = self._draw_finger_debug(frame, fingers_up)

        return frame

    def draw_brush_cursor(
        self,
        frame: np.ndarray,
        tip: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        is_eraser: bool = False,
    ) -> np.ndarray:
        """Draw a preview cursor at the fingertip."""
        if is_eraser:
            cv2.circle(frame, tip, size, (128, 128, 128), 2, cv2.LINE_AA)
            cv2.line(frame, (tip[0]-size, tip[1]), (tip[0]+size, tip[1]), (128,128,128), 1)
            cv2.line(frame, (tip[0], tip[1]-size), (tip[0], tip[1]+size), (128,128,128), 1)
        else:
            cv2.circle(frame, tip, max(size // 2, 3), color, -1, cv2.LINE_AA)
            cv2.circle(frame, tip, max(size // 2, 3), (255,255,255), 1, cv2.LINE_AA)
        return frame

    def draw_help(self, frame: np.ndarray) -> np.ndarray:
        lines = [
            "── GESTURES ──────────────────",
            "☞ Index only       Draw",
            "☞ Open palm        Erase",
            "☞ Peace sign       Clear",
            "☞ Thumb only       Undo",
            "☞ Thumb+Pinky      Redo",
            "☞ 4 fingers        Save",
            "☞ 3 fingers        Brush +",
            "☞ Index+Pinky      Brush -",
            "☞ Pinch            Cycle color",
            "☞ Ring+Pinky       Rectangle",
            "☞ Thumb+Idx+Mid    Circle",
            "",
            "── KEYBOARD ──────────────────",
            "  Q/Esc   Quit",
            "  S       Save PNG",
            "  H       Toggle help",
            "  +/-     Brush size",
            "  [/]     Eraser size",
            "  U/R     Undo/Redo",
            "  1-8     Select colour",
            "  F       Freehand mode",
            "  X       Rectangle mode",
            "  C       Circle mode",
        ]
        pw = 280; ph = len(lines) * 20 + 20
        x0 = self.w - pw - 12
        y0 = max((self.h - ph) // 2, 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+pw, y0+ph), HUD_BG, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0+pw, y0+ph), HUD_ACCENT, 1)

        for i, line in enumerate(lines):
            col = HUD_ACCENT if "──" in line else HUD_TEXT
            cv2.putText(frame, line, (x0+10, y0+18+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        return frame

    def show_status(self, msg: str, duration: float = 1.8) -> None:
        self._status_msg   = msg
        self._status_until = time.time() + duration
        log.debug("Status: %s", msg)

    # ── Palette ─────────────────────────────────────────────────────────────────

    def _draw_palette(self, frame: np.ndarray, active: int) -> np.ndarray:
        cfg  = self._cfg
        sw   = cfg.swatch_size
        sm   = cfg.swatch_margin
        total_w = len(PALETTE) * (sw + sm) - sm
        x0   = (self.w - total_w) // 2
        y0   = 10

        for i, bgr in enumerate(PALETTE):
            x1, y1 = x0 + i*(sw+sm), y0
            x2, y2 = x1+sw, y1+sw
            # Shadow
            cv2.rectangle(frame, (x1+2, y1+2), (x2+2, y2+2), (0,0,0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
            bw  = 3 if i == active else 1
            bc  = HUD_ACCENT if i == active else (70, 70, 70)
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), bc, bw)
            cv2.putText(frame, str(i+1), (x1+2, y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_TEXT, 1)

        # Active colour name
        cv2.putText(frame, PALETTE_NAMES[active],
                    (x0 + len(PALETTE)*(sw+sm) + 8, y0+sw//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE[active], 1, cv2.LINE_AA)
        return frame

    # ── Info panel ──────────────────────────────────────────────────────────────

    def _draw_info_panel(
        self, frame, gesture, confidence, brush, eraser, mode
    ) -> np.ndarray:
        pad = 10; pw = 220; ph = 110
        x0, y0 = pad, self.h - ph - pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+pw, y0+ph), HUD_BG, -1)
        cv2.addWeighted(overlay, get_config().ui.hud_alpha, frame,
                        1 - get_config().ui.hud_alpha, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0+pw, y0+ph), HUD_ACCENT, 1)

        bx, by = x0+10, y0+20
        label  = GESTURE_LABELS.get(gesture, "?")

        # Confidence bar
        if self._cfg.show_confidence:
            bar_w  = int((pw - 20) * confidence)
            bar_col = HUD_OK if confidence >= 0.8 else HUD_WARN
            cv2.rectangle(frame, (bx, by-12), (bx + pw-20, by-4), (50,50,50), -1)
            cv2.rectangle(frame, (bx, by-12), (bx + bar_w, by-4), bar_col, -1)
            cv2.putText(frame, f"{confidence:.0%}", (bx + pw - 50, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, bar_col, 1)

        cv2.putText(frame, f"Gesture : {label}", (bx, by+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Brush   : {brush}px", (bx, by+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Eraser  : {eraser}px", (bx, by+52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mode    : {mode}", (bx, by+72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, HUD_TEXT, 1, cv2.LINE_AA)
        return frame

    # ── FPS ─────────────────────────────────────────────────────────────────────

    def _fps(self) -> int:
        now = time.time()
        return sum(1 for t in self._fps_buf if now - t < 1.0)

    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        fps = self._fps()
        col = HUD_OK if fps >= 20 else HUD_WARN
        cv2.putText(frame, f"FPS {fps}", (self.w - 100, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)
        return frame

    def _draw_fps_graph(self, frame: np.ndarray) -> np.ndarray:
        """Sparkline FPS graph in the bottom-right corner."""
        gw, gh = 100, 36
        x0 = self.w - gw - 10
        y0 = self.h - gh - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+gw, y0+gh), HUD_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        vals = list(self._fps_buf)
        if len(vals) < 2:
            return frame

        # Compute per-second fps at each recent sample
        now = time.time()
        pts = []
        for i, t in enumerate(vals):
            x = x0 + int(gw * i / max(len(vals)-1, 1))
            fps_at = sum(1 for tt in vals if abs(tt - t) < 0.5)
            y = y0 + gh - int(gh * min(fps_at, 60) / 60)
            pts.append((x, y))

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], HUD_ACCENT, 1)
        return frame

    # ── Status ───────────────────────────────────────────────────────────────────

    def _draw_status(self, frame: np.ndarray) -> np.ndarray:
        msg  = self._status_msg
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cx   = (self.w - size[0]) // 2
        cy   = self.h // 2
        cv2.rectangle(frame, (cx-10, cy-30), (cx+size[0]+10, cy+10), HUD_BG, -1)
        cv2.putText(frame, msg, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, HUD_ACCENT, 2, cv2.LINE_AA)
        return frame

    # ── Finger debug ─────────────────────────────────────────────────────────────

    def _draw_finger_debug(self, frame: np.ndarray, fingers: list[bool]) -> np.ndarray:
        labels = ["T", "I", "M", "R", "P"]
        x0, y0 = self.w - 108, self.h - 30
        for i, (l, up) in enumerate(zip(labels, fingers)):
            col = HUD_OK if up else (50, 50, 50)
            cv2.circle(frame, (x0 + i*20, y0), 7, col, -1)
            cv2.putText(frame, l, (x0+i*20-4, y0+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        return frame
