"""
utils/augment_data.py  –  Augment gesture landmark dataset.

Techniques
----------
  mirror     : flip X-axis (simulates opposite handedness)
  noise      : add Gaussian noise (simulates sensor jitter)
  scale      : uniform scale ± 10%  (simulates distance variation)
  rotate_z   : small 2D wrist rotation (simulates hand tilt)

Usage
-----
    python utils/augment_data.py [--factor 3] [--out data/augmented.npz]
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config import get_config
from logger import setup_logging, get_logger

setup_logging()
log = get_logger(__name__)


def mirror(X: np.ndarray) -> np.ndarray:
    """Mirror x-coordinates of all landmarks (shape N,63)."""
    out = X.copy()
    out[:, 0::3] = 1.0 - out[:, 0::3]
    return out


def add_noise(X: np.ndarray, sigma: float = 0.008) -> np.ndarray:
    return np.clip(X + np.random.normal(0, sigma, X.shape), 0, 1).astype(np.float32)


def scale(X: np.ndarray, min_s: float = 0.90, max_s: float = 1.10) -> np.ndarray:
    """Uniform scale relative to wrist (landmark 0)."""
    out = X.copy()
    # wrist x,y,z are indices 0,1,2
    wrist = out[:, :3].copy()
    s = np.random.uniform(min_s, max_s, (len(X), 1))
    for i in range(0, 63, 3):
        out[:, i]   = wrist[:, 0] + s[:, 0] * (out[:, i]   - wrist[:, 0])
        out[:, i+1] = wrist[:, 1] + s[:, 0] * (out[:, i+1] - wrist[:, 1])
    return np.clip(out, 0, 1).astype(np.float32)


def rotate_z(X: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    """Rotate 2D hand pose around wrist."""
    out = X.copy()
    angles = np.random.uniform(-max_deg, max_deg, len(X)) * np.pi / 180
    cx = out[:, 0]; cy = out[:, 1]   # wrist x, y
    for i in range(0, 63, 3):
        dx = out[:, i]   - cx
        dy = out[:, i+1] - cy
        out[:, i]   = cx + dx * np.cos(angles) - dy * np.sin(angles)
        out[:, i+1] = cy + dx * np.sin(angles) + dy * np.cos(angles)
    return np.clip(out, 0, 1).astype(np.float32)


def augment(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Return augmented (X_aug, y_aug) with `factor` extra copies per sample.
    Original data is always included (prepended).
    """
    all_X = [X]
    all_y = [y]

    ops = [
        lambda x: add_noise(x),
        lambda x: scale(x),
        lambda x: mirror(x),
        lambda x: rotate_z(x),
        lambda x: add_noise(scale(x)),
        lambda x: add_noise(mirror(x)),
    ]

    for i in range(factor):
        op  = ops[i % len(ops)]
        aug = op(X)
        all_X.append(aug)
        all_y.append(y)

    X_out = np.vstack(all_X).astype(np.float32)
    y_out = np.concatenate(all_y).astype(np.int64)

    # Shuffle
    idx = np.random.permutation(len(y_out))
    return X_out[idx], y_out[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factor", type=int, default=3, help="Augmentation multiplier")
    p.add_argument("--out",    default=None,         help="Output .npz path")
    args = p.parse_args()

    cfg      = get_config()
    src_path = cfg.paths.dataset
    if not src_path.exists():
        log.critical("Dataset not found at %s", src_path)
        sys.exit(1)

    data       = np.load(src_path)
    X, y       = data["X"], data["y"]
    log.info("Original: %d samples", len(y))

    X_aug, y_aug = augment(X, y, factor=args.factor)
    log.info("Augmented: %d samples (×%d)", len(y_aug), args.factor + 1)

    out = Path(args.out) if args.out else cfg.paths.data_dir / "augmented.npz"
    np.savez(out, X=X_aug, y=y_aug)
    log.info("Saved to %s", out)

if __name__ == "__main__":
    main()
