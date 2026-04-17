"""
utils/export_onnx.py  –  Export GestureNet to ONNX format.

Usage
-----
    python utils/export_onnx.py [--weights models/gesture_net.pt]
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from config import get_config
from logger import setup_logging, get_logger
from gesture_recognizer import GestureNet, INPUT_DIM

setup_logging()
log = get_logger(__name__)


def export(weights_path: Path | None = None, output_path: Path | None = None) -> None:
    cfg = get_config()
    w   = weights_path or cfg.paths.gesture_weights
    out = output_path  or cfg.paths.gesture_onnx

    if not w.exists():
        log.critical("Weights not found at %s", w); sys.exit(1)

    model = GestureNet()
    model.load_state_dict(torch.load(w, map_location="cpu"))
    model.eval()
    log.info("Loaded GestureNet from %s", w)

    dummy = torch.zeros(1, INPUT_DIM)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out),
        input_names  = ["input"],
        output_names = ["logits"],
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version = 14,
    )
    log.info("Exported ONNX model to %s", out)

    # Quick validation
    try:
        import onnxruntime as ort
        sess  = ort.InferenceSession(str(out))
        dummy_np = np.zeros((1, INPUT_DIM), dtype=np.float32)
        result = sess.run(None, {"input": dummy_np})
        log.info("ONNX validation passed. Output shape: %s", result[0].shape)
    except ImportError:
        log.warning("onnxruntime not installed – skipping validation.")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--output",  default=None)
    args = p.parse_args()

    export(
        Path(args.weights) if args.weights else None,
        Path(args.output)  if args.output  else None,
    )

if __name__ == "__main__":
    main()
