"""
utils/collect_data.py  –  12-gesture data collector
Keys: 0-9, a, b → select gesture | SPACE → toggle | Q → quit+save
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse, cv2, numpy as np
from hand_tracker import HandTracker
from gesture_recognizer import Gesture
from config import cfg
from logger import get_logger
log = get_logger(__name__)

KEY_MAP = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"a":10,"b":11}
HINTS   = {
    0:"IDLE – fist", 1:"DRAW – index only", 2:"ERASE – open palm",
    3:"CLEAR – peace ✌", 4:"COLOR_PICK – pinch", 5:"UNDO – thumb 👍",
    6:"SAVE – pinky only", 7:"REDO – hang-loose 🤙", 8:"BRUSH+ – 3 fingers",
    9:"BRUSH- – horns 🤘", 10:"RECTANGLE – L-shape", 11:"CIRCLE – fist/O",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX)
    p.add_argument("--skip",   type=int, default=2)
    p.add_argument("--output", default=str(cfg.DATA_PATH))
    a = p.parse_args()

    tracker = HandTracker(max_hands=1)
    cap     = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    X_list, y_list, label, collecting, fc = [], [], None, False, 0
    print("\nKeys 0-9/a/b → gesture | SPACE → toggle | Q → quit\n")
    for k,v in KEY_MAP.items():
        print(f"  {k} → {HINTS[v]}")

    output_path = Path(a.output)
    while True:
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        hands = tracker.process(frame)
        if hands and collecting and label is not None:
            fc += 1
            if fc % a.skip == 0:
                X_list.append(hands[0].landmarks.flatten())
                y_list.append(label)
            tracker.draw_landmarks(frame, hands[0])

        hint   = HINTS.get(label, "None") if label is not None else "None"
        status = "COLLECTING" if collecting else "PAUSED"
        col    = (0,200,80) if collecting else (0,80,200)
        cv2.putText(frame, hint,             (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(frame, status,           (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"Total: {len(y_list)}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        for i,g in enumerate(Gesture):
            cnt = y_list.count(g.value)
            c   = (0,200,80) if cnt>=200 else (0,150,200) if cnt>=100 else (50,50,200)
            cv2.putText(frame, f"{g.name[:12]:12}{cnt:4}", (w-220,30+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, c, 1)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        ch  = chr(key) if key < 128 else ""
        if key == ord('q'): break
        elif key == ord(' '): collecting = not collecting
        elif ch in KEY_MAP:
            label = KEY_MAP[ch]; collecting = False
            log.info("Label → %s", HINTS[label])

    cap.release(); cv2.destroyAllWindows(); tracker.release()
    if y_list:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            old  = np.load(output_path)
            X_all = np.vstack([old["X"], np.array(X_list, dtype=np.float32)])
            y_all = np.concatenate([old["y"], y_list])
        else:
            X_all, y_all = np.array(X_list, np.float32), np.array(y_list, np.int64)
        np.savez(output_path, X=X_all, y=y_all)
        log.info("Saved %d samples → %s", len(y_all), output_path)
    else:
        log.warning("No data collected.")

if __name__ == "__main__": main()
