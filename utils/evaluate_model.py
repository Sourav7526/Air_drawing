"""
utils/evaluate_model.py  –  Evaluate GestureNet on held-out data.

Prints
------
  Accuracy, Confusion Matrix, Classification Report

Usage
-----
    python utils/evaluate_model.py [--weights models/gesture_net.pt]
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from config import get_config
from logger import setup_logging, get_logger
from gesture_recognizer import GestureNet, Gesture, GESTURE_NAMES

setup_logging()
log = get_logger(__name__)


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   default=None)
    p.add_argument("--val-split", type=float, default=0.20)
    args = p.parse_args()

    cfg          = get_config()
    weights_path = Path(args.weights) if args.weights else cfg.paths.gesture_weights
    data_path    = cfg.paths.dataset

    if not data_path.exists():
        log.critical("Dataset not found: %s", data_path); sys.exit(1)
    if not weights_path.exists():
        log.critical("Weights not found: %s", weights_path); sys.exit(1)

    data   = np.load(data_path)
    X, y   = data["X"], data["y"]
    _, X_val, _, y_val = train_test_split(X, y, test_size=args.val_split,
                                          stratify=y, random_state=42)

    model = GestureNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    log.info("Loaded weights from %s", weights_path)

    with torch.no_grad():
        Xv     = torch.from_numpy(X_val.astype(np.float32))
        preds  = model(Xv).argmax(1).numpy()

    acc    = accuracy_score(y_val, preds)
    labels = [g.name for g in Gesture]

    print(f"\n{'='*60}")
    print(f"  Validation samples : {len(y_val)}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"{'='*60}")

    print_confusion_matrix(confusion_matrix(y_val, preds), labels)

    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=labels, zero_division=0))

if __name__ == "__main__":
    main()
