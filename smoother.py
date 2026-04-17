"""
smoother.py  –  Moving-average tip smoother to reduce hand jitter.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from config import get_config


class TipSmoother:
    """
    Smooth a stream of (x, y) fingertip positions using a sliding window mean.
    Eliminates sensor jitter without introducing noticeable lag.
    """

    def __init__(self, window: int | None = None) -> None:
        w = window or get_config().gesture.smoothing_window
        self._buf: deque[tuple[int,int]] = deque(maxlen=w)

    def smooth(self, pt: tuple[int, int]) -> tuple[int, int]:
        self._buf.append(pt)
        xs = [p[0] for p in self._buf]
        ys = [p[1] for p in self._buf]
        return int(np.mean(xs)), int(np.mean(ys))

    def reset(self) -> None:
        self._buf.clear()
