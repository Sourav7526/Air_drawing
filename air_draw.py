"""
air_draw.py  –  Main application loop for the Air Drawing System.

Pipeline
--------
  Camera → HandTracker (MediaPipe) → GestureRecognizer (PyTorch/rules)
         → Canvas (OpenCV) → UI overlay → Display

Usage
-----
    python air_draw.py [options]

    --camera INT      Camera index (default: 0)
    --width  INT      Frame width  (default: 1280)
    --height INT      Frame height (default: 720)
    --debug           Enable finger debug bar + confidence display
    --no-gpu          Force CPU inference
    --onnx            Use ONNX runtime for gesture inference
    --skip INT        Skip N frames between processing passes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Bootstrap config first so all modules can import it ───────────────────────
from config import get_config
from logger import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)


def _apply_cli(args: argparse.Namespace) -> None:
    cfg = get_config()
    cfg.camera.index       = args.camera
    cfg.camera.width       = args.width
    cfg.camera.height      = args.height
    cfg.camera.skip_frames = args.skip
    if args.no_gpu:
        cfg.model.use_gpu  = False
    if args.onnx:
        cfg.model.use_onnx = True
    if args.debug:
        cfg.ui.show_finger_debug = True
        cfg.ui.show_confidence   = True


def _find_camera(preferred: int) -> int:
    """Try preferred index first, then scan 0-4."""
    for idx in [preferred] + [i for i in range(5) if i != preferred]:
        cap = cv2.VideoCapture(idx)
        ok, _ = cap.read()
        cap.release()
        if ok:
            log.info("Camera found at index %d", idx)
            return idx
    raise RuntimeError("No usable camera found (tried indices 0-4).")


def _open_camera(index: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    log.info("Camera %d opened at %dx%d @ %dfps", index, w, h, fps)
    return cap


def _most_common(lst: list) -> object:
    return max(set(lst), key=lst.count) if lst else None


def _save_drawing(canvas, width: int, height: int) -> Path:
    from config import get_config
    import time as _time
    cfg  = get_config()
    ts   = _time.strftime("%Y%m%d_%H%M%S")
    path = cfg.paths.saved_dir / f"drawing_{ts}.png"
    blank  = np.full((height, width, 3), 255, dtype=np.uint8)
    output = canvas.composite(blank)
    cv2.imwrite(str(path), output)
    log.info("Drawing saved to %s", path)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from hand_tracker       import HandTracker
    from gesture_recognizer import GestureRecognizer, Gesture
    from canvas             import Canvas, DrawMode
    from ui                 import UI
    from smoother           import TipSmoother

    cfg = get_config()
    _apply_cli(args)

    # ── Camera ────────────────────────────────────────────────────────────────
    try:
        cam_idx = _find_camera(cfg.camera.index)
        cap     = _open_camera(cam_idx, cfg.camera.width, cfg.camera.height, cfg.camera.fps)
    except RuntimeError as e:
        log.critical(str(e))
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Components ────────────────────────────────────────────────────────────
    try:
        tracker    = HandTracker()
        recognizer = GestureRecognizer()
    except Exception as e:
        log.critical("Component init failed: %s", e)
        cap.release()
        sys.exit(1)

    canvas   = Canvas(W, H)
    ui       = UI(W, H)
    smoother = TipSmoother()

    # ── State ─────────────────────────────────────────────────────────────────
    WINDOW  = "✏  Air Drawing  –  Press H for help"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)

    show_help     = True
    prev_tip      = None
    prev_gesture  = Gesture.IDLE
    gesture_buf: list[Gesture] = []
    frame_idx     = 0
    debounce      = cfg.gesture.debounce_frames
    jitter        = cfg.gesture.jitter_threshold
    skip          = cfg.camera.skip_frames

    log.info("Air Drawing started. Press Q or Esc to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Camera read failed – retrying.")
            time.sleep(0.05)
            continue

        frame     = cv2.flip(frame, 1)
        frame_idx += 1

        # ── Hand tracking (every frame) ───────────────────────────────────────
        hands      = tracker.process(frame)
        gesture    = Gesture.IDLE
        confidence = 0.0
        fingers_up = None
        tip_raw    = None

        if hands:
            hand       = hands[0]
            fingers_up = hand.fingers_up
            tip_raw    = hand.index_tip_px

            # Skip heavy gesture inference on designated frames
            if skip == 0 or (frame_idx % (skip + 1) == 0):
                gesture    = recognizer.recognize(hand.landmarks, fingers_up)
                confidence = recognizer.last_confidence

            tracker.draw_landmarks(frame, hand)

        # ── Debounce ──────────────────────────────────────────────────────────
        gesture_buf.append(gesture)
        if len(gesture_buf) > debounce:
            gesture_buf.pop(0)
        stable = _most_common(gesture_buf) or Gesture.IDLE

        # ── Smooth tip position ───────────────────────────────────────────────
        tip = smoother.smooth(tip_raw) if tip_raw else None

        # ── Canvas actions ────────────────────────────────────────────────────
        just_changed = (stable != prev_gesture)

        if stable == Gesture.DRAW and tip:
            if prev_gesture != Gesture.DRAW:
                canvas.begin_stroke(*tip)
            else:
                dx, dy = tip[0] - (prev_tip or tip)[0], tip[1] - (prev_tip or tip)[1]
                if dx*dx + dy*dy >= jitter**2:
                    canvas.add_point(*tip)

        elif stable in (Gesture.DRAW_RECTANGLE, Gesture.DRAW_CIRCLE) and tip:
            mode = DrawMode.RECTANGLE if stable == Gesture.DRAW_RECTANGLE else DrawMode.CIRCLE
            canvas.draw_mode = mode
            if prev_gesture != stable:
                canvas.begin_stroke(*tip)
            else:
                frame = canvas.shape_preview_frame(frame, *tip)

        else:
            if prev_gesture in (Gesture.DRAW, Gesture.DRAW_RECTANGLE, Gesture.DRAW_CIRCLE):
                canvas.end_stroke(*(tip or (0, 0)))
                canvas.draw_mode = DrawMode.FREEHAND
            smoother.reset()
            prev_tip = None

            if stable == Gesture.ERASE and tip:
                canvas.erase(*tip)

            elif stable == Gesture.CLEAR and just_changed:
                canvas.clear()
                ui.show_status("Canvas cleared")

            elif stable == Gesture.UNDO and just_changed:
                canvas.undo()
                ui.show_status("Undo")

            elif stable == Gesture.REDO and just_changed:
                canvas.redo()
                ui.show_status("Redo")

            elif stable == Gesture.COLOR_PICK and just_changed:
                canvas.next_color()
                ui.show_status(f"Color: {canvas.color_idx + 1}")

            elif stable == Gesture.SAVE_CANVAS and just_changed:
                p = _save_drawing(canvas, W, H)
                ui.show_status(f"Saved: {p.name}")

            elif stable == Gesture.BRUSH_SIZE_UP and just_changed:
                canvas.increase_thickness()
                ui.show_status(f"Brush: {canvas.thickness}px")

            elif stable == Gesture.BRUSH_SIZE_DOWN and just_changed:
                canvas.decrease_thickness()
                ui.show_status(f"Brush: {canvas.thickness}px")

        if stable == Gesture.DRAW and tip:
            prev_tip = tip

        prev_gesture = stable

        # ── Brush / eraser cursor ─────────────────────────────────────────────
        if tip:
            is_eraser = (stable == Gesture.ERASE)
            size      = canvas.eraser_sz if is_eraser else canvas.thickness
            ui.draw_brush_cursor(frame, tip, canvas.current_color, size, is_eraser)

        # ── Composite canvas then HUD ─────────────────────────────────────────
        frame = canvas.composite(frame)
        frame = ui.render(
            frame,
            current_color_idx = canvas.color_idx,
            current_gesture   = stable,
            confidence        = confidence,
            brush_thickness   = canvas.thickness,
            eraser_size       = canvas.eraser_sz,
            draw_mode         = canvas.draw_mode.name,
            fingers_up        = fingers_up if cfg.ui.show_finger_debug else None,
        )
        if show_help:
            frame = ui.draw_help(frame)

        cv2.imshow(WINDOW, frame)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            p = _save_drawing(canvas, W, H)
            ui.show_status(f"Saved: {p.name}")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('+') or key == ord('='):
            canvas.increase_thickness()
            ui.show_status(f"Brush: {canvas.thickness}px")
        elif key == ord('-'):
            canvas.decrease_thickness()
            ui.show_status(f"Brush: {canvas.thickness}px")
        elif key == ord('['):
            canvas.decrease_eraser()
            ui.show_status(f"Eraser: {canvas.eraser_sz}px")
        elif key == ord(']'):
            canvas.increase_eraser()
            ui.show_status(f"Eraser: {canvas.eraser_sz}px")
        elif key == ord('u'):
            canvas.undo(); ui.show_status("Undo")
        elif key == ord('r'):
            canvas.redo(); ui.show_status("Redo")
        elif key == ord('f'):
            canvas.draw_mode = DrawMode.FREEHAND; ui.show_status("Freehand")
        elif key == ord('x'):
            canvas.draw_mode = DrawMode.RECTANGLE; ui.show_status("Rectangle")
        elif key == ord('c'):
            canvas.draw_mode = DrawMode.CIRCLE; ui.show_status("Circle")
        elif ord('1') <= key <= ord('8'):
            canvas.set_color_idx(int(chr(key)) - 1)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    log.info("Air Drawing exited cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Air Drawing System")
    p.add_argument("--camera", type=int,   default=0,    help="Camera index")
    p.add_argument("--width",  type=int,   default=1280, help="Frame width")
    p.add_argument("--height", type=int,   default=720,  help="Frame height")
    p.add_argument("--skip",   type=int,   default=0,    help="Frame skip count")
    p.add_argument("--debug",  action="store_true",      help="Enable debug overlays")
    p.add_argument("--no-gpu", action="store_true",      help="Force CPU inference")
    p.add_argument("--onnx",   action="store_true",      help="Use ONNX inference")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
