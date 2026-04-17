"""
canvas.py  –  BGRA drawing canvas with undo/redo, shapes, smooth interpolation.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from config import get_config
from logger import get_logger

log = get_logger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),     # Red
    (0, 165, 255),   # Orange
    (0, 255, 255),   # Yellow
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (255, 0, 255),   # Magenta
    (255, 255, 255), # White
    (128, 128, 128), # Grey
]
PALETTE_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue", "Magenta", "White", "Grey"]


class DrawMode(Enum):
    FREEHAND  = auto()
    RECTANGLE = auto()
    CIRCLE    = auto()


@dataclass
class Stroke:
    points:    list[tuple[int, int]] = field(default_factory=list)
    color:     tuple[int, int, int]  = (0, 0, 255)
    thickness: int                   = 6
    mode:      DrawMode              = DrawMode.FREEHAND


class Canvas:
    """
    BGRA canvas layer with:
    - Freehand drawing with smooth Catmull-Rom spline interpolation
    - Rectangle and circle shape tools
    - Adjustable brush and eraser
    - Unlimited undo/redo (capped at max_undo_steps)
    """

    def __init__(self, width: int, height: int) -> None:
        cfg = get_config().brush
        self.w, self.h = width, height

        self._strokes:        list[Stroke]           = []
        self._undo_stack:     deque[list[Stroke]]    = deque(maxlen=cfg.max_undo_steps)
        self._redo_stack:     deque[list[Stroke]]    = deque(maxlen=cfg.max_undo_steps)
        self._current_stroke: Optional[Stroke]       = None
        self._bitmap:         np.ndarray             = self._blank()

        self.color_idx:  int       = 0
        self.thickness:  int       = cfg.default_thickness
        self.eraser_sz:  int       = cfg.default_eraser_sz
        self.draw_mode:  DrawMode  = DrawMode.FREEHAND
        self._shape_start: Optional[tuple[int,int]] = None

        log.debug("Canvas created %dx%d", width, height)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def current_color(self) -> tuple[int, int, int]:
        return PALETTE[self.color_idx]

    # ── Drawing API ─────────────────────────────────────────────────────────────

    def begin_stroke(self, x: int, y: int) -> None:
        if self._current_stroke is not None:
            self._commit_stroke()
        self._current_stroke = Stroke(color=self.current_color,
                                      thickness=self.thickness,
                                      mode=self.draw_mode)
        if self.draw_mode == DrawMode.FREEHAND:
            self._current_stroke.points.append((x, y))
        else:
            self._shape_start = (x, y)

    def add_point(self, x: int, y: int) -> None:
        """Add a point to freehand stroke and render incrementally."""
        if self._current_stroke is None:
            self.begin_stroke(x, y)
            return

        if self.draw_mode != DrawMode.FREEHAND:
            return  # shapes render on end_stroke

        pts = self._current_stroke.points
        if pts:
            # Smooth interpolation: draw a segment with Catmull-Rom if enough pts
            pts.append((x, y))
            if len(pts) >= 4:
                self._draw_catmull_rom(pts[-4], pts[-3], pts[-2], pts[-1])
            elif len(pts) >= 2:
                cv2.line(self._bitmap, pts[-2], pts[-1],
                         (*self._current_stroke.color, 255),
                         self._current_stroke.thickness, cv2.LINE_AA)
        else:
            pts.append((x, y))

    def end_stroke(self, x: int = 0, y: int = 0) -> None:
        """Finish current stroke; for shapes, x/y is the endpoint."""
        if self._current_stroke is None:
            return

        if self.draw_mode == DrawMode.RECTANGLE and self._shape_start:
            self._draw_rect_on_bitmap(self._shape_start, (x, y))
            self._current_stroke.points = [self._shape_start, (x, y)]

        elif self.draw_mode == DrawMode.CIRCLE and self._shape_start:
            r = int(np.linalg.norm(np.array((x, y)) - np.array(self._shape_start)))
            self._draw_circle_on_bitmap(self._shape_start, r)
            self._current_stroke.points = [self._shape_start, (x, y)]

        self._shape_start = None
        self._commit_stroke()

    def erase(self, cx: int, cy: int) -> None:
        self.end_stroke()
        cv2.circle(self._bitmap, (cx, cy), self.eraser_sz, (0, 0, 0, 0), -1)

    def clear(self) -> None:
        self._save_undo()
        self._strokes.clear()
        self._bitmap = self._blank()
        self._current_stroke = None
        log.info("Canvas cleared.")

    def undo(self) -> None:
        if not self._undo_stack:
            log.debug("Nothing to undo.")
            return
        self._redo_stack.append(list(self._strokes))
        self._strokes = self._undo_stack.pop()
        self._current_stroke = None
        self._redraw()
        log.debug("Undo. Strokes remaining: %d", len(self._strokes))

    def redo(self) -> None:
        if not self._redo_stack:
            log.debug("Nothing to redo.")
            return
        self._undo_stack.append(list(self._strokes))
        self._strokes = self._redo_stack.pop()
        self._current_stroke = None
        self._redraw()
        log.debug("Redo. Strokes: %d", len(self._strokes))

    # ── Shape preview (drawn on top of bitmap, not committed) ──────────────────

    def shape_preview_frame(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """Composite a live shape preview onto frame without modifying canvas."""
        if self._shape_start is None or self._current_stroke is None:
            return frame
        preview = frame.copy()
        c = self._current_stroke.color
        t = self._current_stroke.thickness

        if self.draw_mode == DrawMode.RECTANGLE:
            cv2.rectangle(preview, self._shape_start, (x, y), c, t, cv2.LINE_AA)
        elif self.draw_mode == DrawMode.CIRCLE:
            r = int(np.linalg.norm(np.array((x, y)) - np.array(self._shape_start)))
            cv2.circle(preview, self._shape_start, r, c, t, cv2.LINE_AA)
        return preview

    # ── Compositing ─────────────────────────────────────────────────────────────

    def composite(self, bgr_frame: np.ndarray) -> np.ndarray:
        bgra  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA).astype(np.float32)
        canv  = self._bitmap.astype(np.float32)
        alpha = canv[:, :, 3:4] / 255.0
        out   = bgra * (1 - alpha) + canv * alpha
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    # ── Palette helpers ─────────────────────────────────────────────────────────

    def next_color(self)          -> None: self.color_idx = (self.color_idx + 1) % len(PALETTE)
    def prev_color(self)          -> None: self.color_idx = (self.color_idx - 1) % len(PALETTE)
    def set_color_idx(self, i: int) -> None: self.color_idx = i % len(PALETTE)

    def increase_thickness(self) -> None:
        cfg = get_config().brush
        self.thickness = min(self.thickness + cfg.thickness_step, cfg.max_thickness)

    def decrease_thickness(self) -> None:
        cfg = get_config().brush
        self.thickness = max(self.thickness - cfg.thickness_step, cfg.min_thickness)

    def increase_eraser(self) -> None:
        self.eraser_sz = min(self.eraser_sz + get_config().brush.eraser_step, 100)

    def decrease_eraser(self) -> None:
        self.eraser_sz = max(self.eraser_sz - get_config().brush.eraser_step, 10)

    # ── Private ─────────────────────────────────────────────────────────────────

    def _blank(self) -> np.ndarray:
        return np.zeros((self.h, self.w, 4), dtype=np.uint8)

    def _save_undo(self) -> None:
        self._undo_stack.append(list(self._strokes))
        self._redo_stack.clear()

    def _commit_stroke(self) -> None:
        s = self._current_stroke
        if s and s.points:
            self._save_undo()
            self._strokes.append(s)
        self._current_stroke = None

    def _redraw(self) -> None:
        self._bitmap = self._blank()
        for s in self._strokes:
            if s.mode == DrawMode.FREEHAND:
                pts = s.points
                for i in range(1, len(pts)):
                    cv2.line(self._bitmap, pts[i-1], pts[i],
                             (*s.color, 255), s.thickness, cv2.LINE_AA)
            elif s.mode == DrawMode.RECTANGLE and len(s.points) == 2:
                self._draw_rect_on_bitmap(s.points[0], s.points[1], s.color, s.thickness)
            elif s.mode == DrawMode.CIRCLE and len(s.points) == 2:
                r = int(np.linalg.norm(np.array(s.points[1]) - np.array(s.points[0])))
                self._draw_circle_on_bitmap(s.points[0], r, s.color, s.thickness)

    def _draw_rect_on_bitmap(self, p1, p2, color=None, t=None) -> None:
        c = color or self.current_color
        cv2.rectangle(self._bitmap, p1, p2, (*c, 255), t or self.thickness, cv2.LINE_AA)

    def _draw_circle_on_bitmap(self, center, radius, color=None, t=None) -> None:
        c = color or self.current_color
        cv2.circle(self._bitmap, center, radius, (*c, 255), t or self.thickness, cv2.LINE_AA)

    def _draw_catmull_rom(self, p0, p1, p2, p3, steps: int = 8) -> None:
        """Draw a smooth Catmull-Rom spline segment between p1 and p2."""
        def cr(t):
            t2, t3 = t * t, t * t * t
            return (
                0.5 * ((-t3 + 2*t2 - t) * p0[i] + (3*t3 - 5*t2 + 2) * p1[i]
                       + (-3*t3 + 4*t2 + t) * p2[i] + (t3 - t2) * p3[i])
                for i in range(2)
            )
        prev = p1
        for k in range(1, steps + 1):
            t = k / steps
            nx, ny = (int(v) for v in cr(t))
            cv2.line(self._bitmap, prev, (nx, ny),
                     (*self.current_color, 255), self.thickness, cv2.LINE_AA)
            prev = (nx, ny)
